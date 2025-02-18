# Pickling support for contextvars.Context objects
# Copyright (c) 2021  Anselm Kruis
#
# This library is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Suite 500, Boston, MA  02110-1335  USA.

'''
:mod:`cvpickle` --- make :class:`contextvars.Context` picklable

Pickling of :class:`~contextvars.Context` objects is not possible by default for two reasons, given in
https://www.python.org/dev/peps/pep-0567/#making-context-objects-picklable:

   1. ContextVar objects do not have __module__ and __qualname__ attributes,
      making straightforward pickling of Context objects impossible.
   2. Not all context variables refer to picklable objects. Making a ContextVar
      picklable must be an opt-in.

The module :mod:`cvpickle` provides a reducer (class :class:`ContextReducer`) for context objects.
You have to register a ContextVar with the reducer to get it pickled.

For convenience, the module provides a global :class:`ContextReducer` object in
:data:`cvpickle.global_context_reducer` and ContextVar (un-)registration functions
:func:`cvpickle.register_contextvar` and :func:`cvpickle.deregister_contextvar`

A minimal example:

    >>> import cvpickle
    >>> import contextvars
    >>>
    >>> my_context_var = contextvars.ContextVar("my_context_var")
    >>> cvpickle.register_contextvar(my_context_var, __name__)

'''

import contextvars
import copyreg
import importlib
import types
from pickle import _getattribute


class _ContextVarProxy:
    def __init__(self, module_name, qualname):
        self.module_name = module_name
        self.qualname = qualname


def _context_factory(cls, mapping):
    if cls is None:
        context = contextvars.Context()
    else:
        context = cls()

    for (modulename, qualname), value in mapping.items():
        module = importlib.import_module(modulename)
        cv = _getattribute(module, qualname)[0]
        context.run(cv.set, value)
    return context


class ContextReducer:
    """A *ContextReducer* object is a "reduction" function for a :class:`~contextvars.Context` object.

    An *ContextReducer* object knows which context variables can be pickled.
    """

    def __init__(self, *, auto_register=False, factory_is_copy_context=False):
        # contextvars.ContextVar is hashable, but it is not possible to create a weak reference
        # to a ContextVar (as of Python 3.7.1). Therefore we use a regular dictionary instead of
        # weakref.WeakKeyDictionary(). That's no problem, because deleting a ContextVar leaks
        # references anyway
        self.picklable_contextvars = {}

        #: If set to :data:`True`, call :func:`copyreg.pickle` to declare this *ContextReducer* as
        #: "reduction" function for :class:`~contextvars.Context` objects, when the
        #: :meth:`register_contextvar` is called for the first time.
        self.auto_register = auto_register

        #: If set to :data:`True`, use :func:`contextvars.copy_context` to create a new
        #: :class:`~contextvars.Context` object upon unpickling. This way the unpickled
        #: context variables are added to the existing context variables.
        self.factory_is_copy_context = factory_is_copy_context

    def __call__(self, context):
        """Reduce a contextvars.Context object
        """
        if not isinstance(context, contextvars.Context):
            raise TypeError('Argument must be a Context object not {}'.format(type(context).__name__))
        cvars = {}
        for cv, value in context.items():
            mod_and_name = self.picklable_contextvars.get(cv)
            if mod_and_name is not None:
                cvars[mod_and_name] = value

        if self.factory_is_copy_context:
            cls = contextvars.copy_context
        else:
            cls = type(context)
            if cls is contextvars.Context:
                # class contextvars.Context can't be pickled, because its __module__ is 'builtins' (Python 3.7.5)
                cls = None
        return _context_factory, (cls, cvars)

    def register_contextvar(self, contextvar, module, qualname=None, *, validate=True):
        """Register *contextvar* with this :class:`ContextReducer`

        Declare, that the context variable *contextvar* can be pickled.

        :param contextvar: a context variable
        :type contextvar: :class:`~contextvars.ContextVar`
        :param module: the module object or the module name, where *contextvar* is declared
        :type module: :class:`~types.ModuleType` or :class:`str`
        :param qualname: the qualified name of *contextvar* in *module*. If unset, *contextvar.name* is used.
        :type qualname: :class:`str`
        :param validate: if true, check that *contextvar* can be accessed as *module.qualname*.
        :type validate: :class:`boolean`
        :raises TypeError: if *contextvar* is not an instance of :class:`~contextvars.ContextVar`
        :raises ValueError: if *contextvar* is not *module.qualname*.
        """
        if not isinstance(contextvar, contextvars.ContextVar):
            raise TypeError('Argument 1 must be a ContextVar object not {}'.format(type(contextvar).__name__))

        modulename = module
        is_module = isinstance(module, types.ModuleType)
        if is_module:
            modulename = module.__name__
        if qualname is None:
            qualname = contextvar.name
        if validate:
            if not is_module:
                module = importlib.import_module(modulename)
            v = _getattribute(module, qualname)[0]  # raises AttributeError
            if v is not contextvar:
                raise ValueError('Not the same object: ContextVar {} and global {}.{}'.format(contextvar.name, modulename, qualname))
        self.picklable_contextvars[contextvar] = (modulename, qualname)
        if self.auto_register:
            self.auto_register = False
            copyreg.pickle(contextvars.Context, self)
            # in case of stackless python enable context pickling
            try:
                from stackless import PICKLEFLAGS_PICKLE_CONTEXT, pickle_flags, pickle_flags_default
            except ImportError:
                pass
            else:
                pickle_flags(PICKLEFLAGS_PICKLE_CONTEXT, PICKLEFLAGS_PICKLE_CONTEXT)
                pickle_flags_default(PICKLEFLAGS_PICKLE_CONTEXT, PICKLEFLAGS_PICKLE_CONTEXT)

    def deregister_contextvar(self, contextvar):
        """Deregister *contextvar* from this :class:`ContextReducer`

        Declare, that the context variable *contextvar* can't be pickled.

        :param contextvar: a context variable
        :type contextvar: :class:`~contextvars.ContextVar`
        :raises KeyError: if *contextvar* hasn't been registered.
        """
        del self.picklable_contextvars[contextvar]


#: A global :class:`ContextReducer` object.
#:
#: The attributes are set as follows
#:
#:  * :attr:`~ContextReducer.auto_register`: :data:`True`
#:  * :attr:`~ContextReducer.factory_is_copy_context`: :data:`True`
#:
#: :meta hide-value:
#:
global_context_reducer = ContextReducer(auto_register=True, factory_is_copy_context=True)


def register_contextvar(contextvar, module, qualname=None, *, validate=True):
    """Register *contextvar* with :data:`global_context_reducer`

    See :meth:`ContextReducer.register_contextvar`.
    """
    return global_context_reducer.register_contextvar(contextvar, module, qualname, validate=validate)


def deregister_contextvar(contextvar):
    """Deregister *contextvar* from :data:`global_context_reducer`

    See :meth:`ContextReducer.deregister_contextvar`.
    """
    return global_context_reducer.deregister_contextvar(contextvar)
