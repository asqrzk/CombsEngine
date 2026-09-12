# Baselines — what a known machine measured, so a stranger has a ruler

A number without a machine beside it cannot be argued with. These are
records from boxes that have actually sat through the kit, committed so
that anyone running `combs-kit model` on their own hardware can say "this
box is 0.6x the reference" instead of guessing, and so that an outside
reader gets a figure with a date and a build stamp rather than a claim.

Naming: `<machine>-<build>-<seat>.json`.

Rules, and they are the point:

- A baseline is a RECORD, not a hand-typed number. It carries the
  machine, the power source, the build, and how every figure is known.
- Never edit one by hand. Re-measure and replace it, or add a new one.
- `combs-kit diff <baseline> <yours>` refuses to compare across machines,
  because that would report on the desks rather than on the change. It
  allows a cross-build comparison and labels it loudly, since choosing
  between two builds is the live question here and is not a regression
  check.
- A baseline taken on battery is worth less than no baseline. The record
  says which, and the diff shouts when they differ.

The first two are this desk: the same seat on the f32 build, and on the
f16 build with the f16-storage door open. The pair is the estate's build
question in two files.
