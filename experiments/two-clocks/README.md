# Two clocks under load

This experiment asks a narrow question: when a streaming service gets slower, did
generation slow down, or did requests wait longer before generation began?

The recorded comparison holds the workload and deployment fixed while concurrency
moves from 400 to 600 users. The first-token wait rises from 1,154 ms to 4,894 ms,
while the gap between generated tokens moves only from 5.7 ms to 5.9 ms. Throughput
falls from 16,187 to 13,412 streamed SSE chunks per second. Those are two different
clocks and two different observations.

The 450-user run is kept in the data because it does not support a knee: its changes
remain inside the recorded repeat bands. A separate run at 600 users with a raised
admission limit performs worse, but that does not prove a universal admission policy.

## Run the check

```bash
python3 experiments/two-clocks/check.py
```

The checker validates arithmetic and the stated uncertainty boundaries using the
committed observations. It does not rerun a GPU serving campaign. Throughput is a
proxy counted as streamed SSE chunks per second, not tokenizer-exact tokens.

## Evidence boundary

The measurements are recorded from one deployment and workload. The conclusion that
the added delay is consistent with queueing is derived from the two clocks and the
admission-limit comparison; no field here directly measures time spent in a queue.
