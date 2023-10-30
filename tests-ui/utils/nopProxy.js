export const nop = new Proxy(function () {}, {
	get: () => nop,
	set: () => true,
	apply: () => nop,
	construct: () => nop,
});
