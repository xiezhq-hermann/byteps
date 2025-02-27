from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import byteps.torch as bps
import timeit
import numpy as np
import os, sys
from torch.profiler import profile, ProfilerActivity
from torch.cuda import synchronize

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Synthetic Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--fp16-pushpull",
    action="store_true",
    default=False,
    help="use fp16 compression during byteps pushpull",
)

parser.add_argument("--model", type=str, default="resnet50", help="model to benchmark")
parser.add_argument("--batch-size", type=int, default=32, help="input batch size")

parser.add_argument(
    "--num-warmup-batches",
    type=int,
    default=10,
    help="number of warm-up batches that don't count towards benchmark",
)
parser.add_argument(
    "--num-batches-per-iter",
    type=int,
    default=3,
    help="number of batches per benchmark iteration",
)
parser.add_argument(
    "--num-iters", type=int, default=10, help="number of benchmark iterations"
)
parser.add_argument("--num-classes", type=int, default=1000, help="number of classes")

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--profiler", action="store_true", default=False, help="disables profiler"
)
parser.add_argument("--partition", type=int, default=None, help="partition size")
parser.add_argument(
    "--synchronize", action="store_true", default=False, help="disables synchronize"
)
parser.add_argument(
    "--throughput", action="store_true", default=False, help="disables throughput"
)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

bps.init()

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())

cudnn.benchmark = True

if args.model == "speech":
    from torchaudio import models

    model = models.DeepSpeech(256)
elif args.model == "transformer":
    model = torch.nn.Transformer(batch_first=True)
elif args.model == "yolov3":
    from yolov3_model.yolov3 import Yolov3
    model = Yolov3()
else:
    from torchvision import models

    # Set up standard model.
    model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# BytePS: (optional) compression algorithm.
compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(), compression=compression
)

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fake data
datasets = []
for _ in range(100):
    if args.model == "speech":
        data = (torch.rand(args.batch_size, 300, 256),)
        target = torch.LongTensor(args.batch_size, 40).random_() % 40
    elif args.model == "transformer":
        data = (
            torch.rand(args.batch_size, 100, 512),
            torch.rand(args.batch_size, 100, 512),
        )
        target = torch.LongTensor(args.batch_size, 512).random_() % 100
    elif args.model == "yolov3":
        data = (torch.rand(args.batch_size, 3, 448, 448),)
        label_sbbox = torch.rand(args.batch_size, 56, 56, 3, 26)
        label_mbbox = torch.rand(args.batch_size, 28, 28, 3, 26)
        label_lbbox = torch.rand(args.batch_size, 14, 14, 3, 26)
        sbboxes = torch.rand(args.batch_size, 150, 4)
        mbboxes = torch.rand(args.batch_size, 150, 4)
        lbboxes = torch.rand(args.batch_size, 150, 4)
        target = (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    else:
        data = (torch.rand(args.batch_size, 3, 224, 224),)
        target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = tuple(x.cuda() for x in data), tuple(x.cuda() for x in target) if isinstance(target, tuple) else target.cuda()
    datasets.append(data)
data_index = 0


def benchmark_step():
    global data_index

    data = datasets[data_index % len(datasets)]
    data_index += 1
    optimizer.zero_grad()
    output = model(*data)

    if args.model == "yolov3":
        from yolov3_model.loss.yolo_loss import YoloV3Loss
        from yolov3_model.yolov3 import MODEL
        p, p_d = output
        loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(MODEL["ANCHORS"], MODEL["STRIDES"])(p, p_d, *target)
    else:
        # Todo: loss functions for speech and language models
        loss = F.cross_entropy(output, target)
        
    loss.backward()
    optimizer.step()
    if args.synchronize:
        synchronize()


def log(s, nl=True):
    if bps.local_rank() != 0:
        return
    print(s, end="\n" if nl else "")
    sys.stdout.flush()


log("Model: %s" % args.model)
log("Batch size: %d" % args.batch_size)
device = "GPU" if args.cuda else "CPU"
log("Number of %ss: %d" % (device, bps.size()))

# Warm-up
log("Running warmup...")
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log("Running benchmark...")
img_secs = []


def bench_all():
    for x in range(args.num_iters):
        if args.synchronize:
            synchronize()
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log("Iter #%d: %.1f img/sec per %s" % (x, img_sec, device))
        img_secs.append(img_sec)


if args.profiler & (bps.rank() == 0):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        bench_all()
    prof.export_chrome_trace(
        "trace_{}_{}x{}x{}.json".format(
            args.model, args.batch_size, args.num_batches_per_iter, args.num_iters
        )
    )
else:
    bench_all()

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log("Img/sec per %s: %.1f +-%.1f" % (device, img_sec_mean, img_sec_conf))
log(
    "Total img/sec on %d %s(s): %.1f +-%.1f"
    % (bps.size(), device, bps.size() * img_sec_mean, bps.size() * img_sec_conf)
)

if args.throughput & (bps.rank() == 0):
    import xlsxwriter

    with xlsxwriter.Workbook(
        "throughput_{}_{}x{}x{}.xlsx".format(
            args.model, args.batch_size, args.num_batches_per_iter, args.num_iters
        )
    ) as workbook:
        worksheet = workbook.add_worksheet()
        p = 1 / len(img_secs)
        for row_num, data in enumerate(img_secs):
            worksheet.write(row_num, 0, data)
            worksheet.write(row_num, 1, (row_num + 1) * p)
