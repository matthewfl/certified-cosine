#ifndef FE_ROUNDING_MODE
#define FE_ROUNDING_MODE

// include this as a header instead of using glibc so that the compiler can
// inline these methods.
//
// Considering that interval arithmetic is required to /guarantee/ that there
// aren't issues due to floating point rounding, this gets called a lot when it
// is trying to construct a proof.  The overhead of indirecting through a shared
// library function was noticeable

#if __x86_64__

// copied (and modified) from glibc source version 2.19

static inline int
inlined_fesetround (int round)
{
  unsigned short int cw;
  int mxcsr;

  if ((round & ~0xc00) != 0)
    /* ROUND is no valid rounding mode.  */
    return 1;

  /* First set the x87 FPU.  */
  asm ("fnstcw %0" : "=m" (*&cw));
  cw &= ~0xc00;
  cw |= round;
  asm ("fldcw %0" : : "m" (*&cw));

  /* And now the MSCSR register for SSE, the precision is at different bit
     positions in the different units, we need to shift it 3 bits.  */
  asm ("stmxcsr %0" : "=m" (*&mxcsr));
  mxcsr &= ~ 0x6000;
  mxcsr |= round << 3;
  asm ("ldmxcsr %0" : : "m" (*&mxcsr));

  return 0;
}


static inline int
inlined_fegetround (void)
{
  int cw;
  /* We only check the x87 FPU unit.  The SSE unit should be the same
     - and if it's not the same there's no way to signal it.  */

  __asm__ ("fnstcw %0" : "=m" (*&cw));

  return cw & 0xc00;
}


// ensure that we don't conflict with the builtin definition
#include <fenv.h>
#define fesetround inlined_fesetround
#define fegetround inlined_fegetround

#else

#include <fenv.h>

#endif
#endif
