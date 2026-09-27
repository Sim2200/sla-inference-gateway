# Load generator with a 2,000-image replay set baked in (the full ImageNetV2 set is 1.2 GB).
FROM slagw-loadgen
COPY loadtest /work/loadtest
COPY models/dataset.py /work/models/dataset.py
COPY data/replay /work/data/replay
