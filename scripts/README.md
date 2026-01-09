# Scripts

## .gdbinit

Copy `.gdbinit` or `.gdbinit-print-opencog` to `~/.gdbinit`
(That is, to your home directory.)

## util

Contains scripts related to cogutil, under the `util` folder, in
particular for manipulating log files, such as filtering according to
component name or log level, as well as sorting according to some
given order.


How to use Valgrind suppressions
--------------------------------
Using valgrind with opencog can be tricky, because many spurious messages
are created.  These can be suppressed with suppression files, like so:
```
valgrind --suppressions=../scripts/valgrind.guile.suppressions      \
         --suppressions=../scripts/valgrind.python.suppressions     \
         --suppressions=../scripts/valgrind.boost.suppressions      \
         --suppressions=../scripts/valgrind.logger.suppressions     \
         --suppressions=../scripts/valgrind.link-grammar.suppressions  \
    <program_to_debug>
```
where `<program_to_debug>` is probably `opencog/server/cogserver`

Memory leaks:      `valgrind --leak-check=full`

Memory non-leaks:  `valgrind --leak-check=full --show-leak-kinds=all`

Memory profile:    `valgrind --tool=massif`

CPU profile:       `valgrind --tool=callgrind`

then: `callgrind_annotate callgrind.out.nnnn`
