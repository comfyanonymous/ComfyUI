<template>
  <div
    class="overflow-hidden [container-type:inline-size] [mask-image:linear-gradient(to_right,black_70%,transparent)] motion-safe:group-hover:[mask-image:none]"
  >
    <span
      class="whitespace-nowrap inline-block min-w-full text-center [--_marquee-end:min(calc(-100%+100cqw),0px)] motion-safe:group-hover:[animation:marquee-scroll_3s_linear_infinite_alternate]"
    >
      <slot />
    </span>
  </div>
</template>

<style>
@keyframes marquee-scroll {
  0%,
  20% {
    transform: translateX(0);
  }
  80%,
  100% {
    transform: translateX(var(--_marquee-end));
  }
}
</style>
