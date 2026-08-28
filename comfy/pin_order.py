from __future__ import annotations

import weakref


class NullPinOrder:
    enabled = False
    budget_checked = False
    current = -1

    def advance(self):
        self.current += 1

    def protected_indices(self):
        return []

    def close(self):
        pass


class PrefetchPinOrder:
    enabled = True

    def __init__(self, units, window=3):
        self.window = window
        self.current = -1
        self.budget_checked = False
        self.positions = weakref.WeakKeyDictionary()
        self.modules = []
        for index, unit in enumerate(units):
            roots = unit if isinstance(unit, (list, tuple)) else (unit,)
            modules = []
            for root in roots:
                for module in root.modules():
                    if hasattr(module, "_v"):
                        sources = [module]
                        for param_key in ("weight", "bias"):
                            lowvram_source = getattr(module, param_key + "_lowvram_function", None)
                            if lowvram_source is not None:
                                sources.append(lowvram_source)
                        for source in sources:
                            modules.append(source)
                            self.positions[source] = index
                            source._pin_prefetch_order = weakref.ref(self)
            self.modules.append(modules)

    def advance(self):
        self.current += 1
        self.budget_checked = False

    def state(self, module):
        index = self.positions.get(module)
        if index is None or self.current < 0:
            return None
        if index >= self.current:
            distance = index - self.current
            protected = distance < self.window
        else:
            distance = len(self.modules) + index
            protected = False
        return protected, distance

    def protected_indices(self):
        if self.current < 0:
            return []
        return list(range(self.current, min(len(self.modules), self.current + self.window)))

    def close(self):
        for module in self.positions:
            order = getattr(module, "_pin_prefetch_order", None)
            if order is not None and order() is self:
                del module._pin_prefetch_order
        self.positions.clear()
        self.modules.clear()


def prefetch_pin_state(module):
    order = getattr(module, "_pin_prefetch_order", None)
    order = None if order is None else order()
    return None if order is None else order.state(module)


def prefetch_budget_checked(module):
    order = getattr(module, "_pin_prefetch_order", None)
    order = None if order is None else order()
    return order is not None and order.budget_checked
