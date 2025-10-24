from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
import os
import typing
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, Optional, Iterable

from .platform_path import construct_path

supported_pt_extensions = frozenset(['.ckpt', '.pt', '.pt2', '.bin', '.pth', '.safetensors', '.pkl', '.sft' ".index.json", ".gguf"])
extension_mimetypes_cache = {
    "webp": "image",
    "fbx": "model",
}

logger = logging.getLogger(__name__)


def do_add(collection: list | set, index: int | None, item: Any):
    if isinstance(collection, list) and index == 0:
        collection.insert(0, item)
    elif isinstance(collection, set):
        collection.add(item)
    else:
        assert isinstance(collection, list)
        collection.append(item)


class FolderPathsTuple:
    def __init__(self, folder_name: str = None, paths: list[str] = None, supported_extensions: set[str] = None, parent: Optional[weakref.ReferenceType[FolderNames]] = None):
        paths = paths or []
        supported_extensions = supported_extensions or set(supported_pt_extensions)

        self.folder_name = folder_name
        self.parent = parent
        self._paths = paths
        self._supported_extensions = supported_extensions

    @property
    def supported_extensions(self) -> set[str] | SupportedExtensions:
        if self.parent is not None and self.folder_name is not None:
            return SupportedExtensions(self.folder_name, self.parent)
        else:
            return self._supported_extensions

    @supported_extensions.setter
    def supported_extensions(self, value: set[str] | SupportedExtensions):
        self.supported_extensions.clear()
        self.supported_extensions.update(value)

    @property
    def paths(self) -> list[str] | PathsList:
        if self.parent is not None and self.folder_name is not None:
            return PathsList(self.folder_name, self.parent)
        else:
            return self._paths

    def __iter__(self) -> typing.Generator[typing.Iterable[str]]:
        """
        allows this proxy to behave like a tuple everywhere it is used by the custom nodes
        :return:
        """
        yield self.paths
        yield self.supported_extensions

    def __getitem__(self, item: Any):
        if item == 0:
            return self.paths
        if item == 1:
            return self.supported_extensions

        raise RuntimeError("unsupported tuple index")

    def __iadd__(self, other: FolderPathsTuple):
        for path in other.paths:
            self.paths.append(path)

        for ext in other.supported_extensions:
            self.supported_extensions.add(ext)


@dataclasses.dataclass
class PathsList:
    folder_name: str
    parent: weakref.ReferenceType[FolderNames]

    def __iter__(self) -> typing.Generator[str]:
        p: FolderNames = self.parent()
        for path in p.directory_paths(self.folder_name):
            try:
                yield str(path.resolve())
            except (OSError, AttributeError):
                yield str(path)

    def __getitem__(self, item: int):
        paths = [x for x in self]
        return paths[item]

    def append(self, path_str: str | Path):
        p: FolderNames = self.parent()
        p.add_paths(self.folder_name, [path_str])

    def insert(self, index: int, path_str: str | Path):
        p: FolderNames = self.parent()
        p.add_paths(self.folder_name, [path_str], index=index)

    def index(self, value: str | Path):
        value = construct_path(value)
        p: FolderNames = self.parent()
        return [path for path in p.directory_paths(self.folder_name)].index(value)

    def remove(self, value: str | Path):
        p: FolderNames = self.parent()
        p.remove_paths(self.folder_name, [value])

    def __len__(self):
        p: FolderNames = self.parent()
        return len(list(p.directory_paths(self.folder_name)))


@dataclasses.dataclass
class SupportedExtensions:
    folder_name: str
    parent: weakref.ReferenceType[FolderNames]

    def _append_any(self, other):
        if other is None:
            return

        p: FolderNames = self.parent()
        if isinstance(other, str):
            other = {other}
        p.add_supported_extension(self.folder_name, *other)

    def __iter__(self) -> typing.Generator[str]:
        p: FolderNames = self.parent()
        for ext in p.supported_extensions(self.folder_name):
            yield ext

    def clear(self):
        p: FolderNames = self.parent()
        p.remove_all_supported_extensions(self.folder_name)

    def __len__(self):
        p: FolderNames = self.parent()
        return len(list(p.supported_extensions(self.folder_name)))

    def __or__(self, other):
        self._append_any(other)
        return self

    __ior__ = _append_any
    add = _append_any
    update = _append_any


@dataclasses.dataclass
class ApplicationPaths:
    output_directory: Path = dataclasses.field(default_factory=lambda: construct_path("output"))
    temp_directory: Path = dataclasses.field(default_factory=lambda: construct_path("temp"))
    input_directory: Path = dataclasses.field(default_factory=lambda: construct_path("input"))
    user_directory: Path = dataclasses.field(default_factory=lambda: construct_path("user"))

    def __iter__(self) -> typing.Generator[Path]:
        yield self.output_directory
        yield self.temp_directory
        yield self.input_directory
        yield self.user_directory


@dataclasses.dataclass
class AbstractPaths(ABC):
    folder_names: list[str]
    supported_extensions: set[str] = dataclasses.field(default_factory=lambda: set(supported_pt_extensions))

    @abstractmethod
    def directory_paths(self, base_paths: Iterable[Path]) -> typing.Generator[Path]:
        """Generate directory paths based on the given base paths."""
        pass

    @abstractmethod
    def file_paths(self, base_paths: Iterable[Path], relative=False) -> typing.Generator[Path]:
        """Generate file paths based on the given base paths."""
        pass

    @abstractmethod
    def has_folder_name(self, folder_name: str) -> bool:
        """Check if the given folder name is in folder_names."""
        pass

    @abstractmethod
    def remove_path(self, path: str | Path) -> int:
        """
        removes a path
        :param path: the path
        :return: the number of paths removed
        """
        pass


@dataclasses.dataclass
class ModelPaths(AbstractPaths):
    folder_name_base_path_subdir: Path = dataclasses.field(default_factory=lambda: construct_path("models"))
    additional_relative_directory_paths: list[Path] = dataclasses.field(default_factory=list)
    additional_absolute_directory_paths: list[str | Path] = dataclasses.field(default_factory=list)
    folder_names_are_relative_directory_paths_too: bool = dataclasses.field(default_factory=lambda: True)

    def directory_paths(self, base_paths: Iterable[Path]) -> typing.Generator[Path]:
        yielded_so_far: set[Path] = set()
        for base_path in base_paths:
            if self.folder_names_are_relative_directory_paths_too:
                for folder_name in self.folder_names:
                    resolved_default_folder_name_path = base_path / self.folder_name_base_path_subdir / folder_name
                    if not resolved_default_folder_name_path in yielded_so_far:
                        yield resolved_default_folder_name_path
                        yielded_so_far.add(resolved_default_folder_name_path)

            for additional_relative_directory_path in self.additional_relative_directory_paths:
                resolved_additional_relative_path = base_path / additional_relative_directory_path
                if not resolved_additional_relative_path in yielded_so_far:
                    yield resolved_additional_relative_path
                    yielded_so_far.add(resolved_additional_relative_path)

        # resolve all paths
        yielded_so_far = {path.resolve() for path in yielded_so_far}
        for additional_absolute_path in self.additional_absolute_directory_paths:
            try:
                resolved_absolute_path = additional_absolute_path.resolve()
            except (OSError, AttributeError):
                resolved_absolute_path = additional_absolute_path
            if not resolved_absolute_path in yielded_so_far:
                yield resolved_absolute_path
                yielded_so_far.add(resolved_absolute_path)

    def file_paths(self, base_paths: Iterable[Path], relative=False) -> typing.Generator[Path]:
        for path in self.directory_paths(base_paths):
            for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
                if '.git' in dirnames:
                    dirnames.remove('.git')

                for filename in filenames:
                    if any(filename.endswith(ext) for ext in self.supported_extensions):
                        result_path = construct_path(dirpath) / filename
                        if relative:
                            yield result_path.relative_to(path)
                        else:
                            yield result_path

    def has_folder_name(self, folder_name: str) -> bool:
        return folder_name in self.folder_names

    def remove_path(self, path: str | Path) -> int:
        total = 0
        path = construct_path(path)
        for paths_list in (self.additional_absolute_directory_paths, self.additional_relative_directory_paths):
            try:
                while True:
                    paths_list.remove(path)
                    total += 1
            except ValueError:
                pass

        return total


@dataclasses.dataclass
class FolderNames:
    application_paths: typing.Optional[ApplicationPaths] = dataclasses.field(default_factory=ApplicationPaths)
    contents: list[AbstractPaths] = dataclasses.field(default_factory=list)
    base_paths: list[Path] = dataclasses.field(default_factory=list)
    is_root: bool = dataclasses.field(default=lambda: False)

    def supported_extensions(self, folder_name: str) -> typing.Generator[str]:
        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                for supported_extensions in candidate.supported_extensions:
                    for supported_extension in supported_extensions:
                        yield supported_extension

    def directory_paths(self, folder_name: str) -> typing.Generator[Path]:
        for directory_path in itertools.chain.from_iterable([candidate.directory_paths(self.base_paths)
                                                             for candidate in self.contents if candidate.has_folder_name(folder_name)]):
            yield directory_path

    def file_paths(self, folder_name: str, relative=False) -> typing.Generator[Path]:
        for candidate in self.contents:
            if not candidate.has_folder_name(folder_name):
                continue

            for file_path in candidate.file_paths(self.base_paths, relative=relative):
                yield file_path

    def first_existing_or_none(self, folder_name: str, relative_file_path: Path) -> Optional[Path]:
        for candidate in self.contents:
            if not candidate.has_folder_name(folder_name):
                continue
            for directory_path in candidate.directory_paths(self.base_paths):
                candidate_file_path: Path = construct_path(directory_path / relative_file_path)
                try:
                    # todo: this should follow the symlink
                    if Path.exists(candidate_file_path):
                        return candidate_file_path
                except OSError:
                    continue
        return None

    def add_supported_extension(self, folder_name: str, *supported_extensions: str | None):
        if supported_extensions is None:
            return

        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                candidate.supported_extensions.update(supported_extensions)

    def remove_all_supported_extensions(self, folder_name: str):
        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                candidate.supported_extensions.clear()

    def add(self, model_paths: AbstractPaths):
        self.contents.append(model_paths)

    def add_base_path(self, base_path: Path):
        if base_path not in self.base_paths:
            self.base_paths.append(base_path)

    @staticmethod
    def from_dict(folder_paths_dict: dict[str, tuple[typing.Iterable[str], Iterable[str]]] = None) -> FolderNames:
        """
        Turns a dictionary of
        {
          "folder_name": (["folder/paths"], {".supported.extensions"})
        }

        into a FolderNames object
        :param folder_paths_dict: A dictionary
        :return: A FolderNames object
        """
        fn = FolderNames()
        for folder_name, (paths, extensions) in folder_paths_dict.items():
            fn.add(
                ModelPaths(folder_names=[folder_name],
                           supported_extensions=set(extensions),
                           additional_relative_directory_paths=[path for path in paths if not Path(path).is_absolute()],
                           additional_absolute_directory_paths=[path for path in paths if Path(path).is_absolute()], folder_names_are_relative_directory_paths_too=False
                           ))
        return fn

    def __getitem__(self, folder_name) -> FolderPathsTuple:
        if not isinstance(folder_name, str) or folder_name is None:
            raise RuntimeError("expected folder path")
        # todo: it is probably never the intention to do this
        try:
            path = Path(folder_name)
            if path.is_absolute():
                folder_name = path.stem
        except Exception:
            pass
        if not any(candidate.has_folder_name(folder_name) for candidate in self.contents):
            self.add(ModelPaths(folder_names=[folder_name], folder_name_base_path_subdir=construct_path(), supported_extensions=set(), folder_names_are_relative_directory_paths_too=False))
        return FolderPathsTuple(folder_name, parent=weakref.ref(self))

    def add_paths(self, folder_name: str, paths: list[Path | str], index: Optional[int] = None):
        """
        Adds, but does not create, new model paths
        :param folder_name:
        :param paths:
        :param index:
        :return:
        """
        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                self._modify_model_paths(folder_name, paths, set(), candidate, index=index)

    def remove_paths(self, folder_name: str, paths: list[Path | str]):
        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                for path in paths:
                    candidate.remove_path(path)

    def get_paths(self, folder_name: str) -> typing.Generator[AbstractPaths]:
        for candidate in self.contents:
            if candidate.has_folder_name(folder_name):
                yield candidate

    def _modify_model_paths(self, key: str, paths: Iterable[Path | str], supported_extensions: set[str], model_paths: AbstractPaths = None, index: Optional[int] = None) -> AbstractPaths:
        model_paths = model_paths or ModelPaths([key],
                                                supported_extensions=set(supported_extensions),
                                                folder_names_are_relative_directory_paths_too=False)
        if index is not None and index != 0:
            raise ValueError(f"index was {index} but only 0 or None is supported")

        did_add = False
        for path in paths:
            if isinstance(path, str):
                path = construct_path(path)
            # when given absolute paths, try to formulate them as relative paths anyway
            if path.is_absolute():
                for base_path in self.base_paths:
                    did_add = False
                    try:
                        relative_to_basepath = path.relative_to(base_path)
                        potential_folder_name = relative_to_basepath.stem
                        potential_subdir = relative_to_basepath.parent

                        # does the folder_name so far match the key?
                        # and have we not seen this folder before?
                        folder_name_not_already_in_contents = all(not candidate.has_folder_name(potential_folder_name) for candidate in self.contents)
                        if potential_folder_name == key and folder_name_not_already_in_contents:
                            # fix the subdir
                            model_paths.folder_name_base_path_subdir = potential_subdir
                            model_paths.folder_names_are_relative_directory_paths_too = True
                            if folder_name_not_already_in_contents:
                                do_add(model_paths.folder_names, index, potential_folder_name)
                                did_add = True
                        else:
                            # if the folder name doesn't match the key, check if we have ever seen a folder name that matches the key:
                            if model_paths.folder_names_are_relative_directory_paths_too:
                                if potential_subdir == model_paths.folder_name_base_path_subdir:
                                    # then we want to add this to the folder name, because it's probably compatibility
                                    do_add(model_paths.folder_names, index, potential_folder_name)
                                    did_add = True
                                else:
                                    do_add(model_paths.additional_relative_directory_paths, index, relative_to_basepath)
                                    did_add = True
                            else:
                                do_add(model_paths.additional_relative_directory_paths, index, relative_to_basepath)
                                for resolve_folder_name in model_paths.folder_names:
                                    do_add(model_paths.additional_relative_directory_paths, index, model_paths.folder_name_base_path_subdir / resolve_folder_name)
                                    did_add = True

                        # since this was an absolute path that was a subdirectory of one of the base paths,
                        # we are done
                        break
                    except ValueError:
                        # this is not a subpath of the base path
                        pass

                # if we got this far, none of the absolute paths were subdirectories of any base paths
                # add it to our absolute paths
                if not did_add:
                    do_add(model_paths.additional_absolute_directory_paths, index, path)
            else:
                # since this is a relative path, peacefully add it to model_paths
                potential_folder_name = path.stem

                try:
                    relative_to_current_subdir = path.relative_to(model_paths.folder_name_base_path_subdir)

                    # if relative to the current subdir, we observe only one part, we're good to go
                    if len(relative_to_current_subdir.parts) == 1:
                        if potential_folder_name == key:
                            model_paths.folder_names_are_relative_directory_paths_too = True
                        else:
                            # if there already exists a folder_name by this name, do not add it, and switch to all relative paths
                            if any(candidate.has_folder_name(potential_folder_name) for candidate in self.contents):
                                model_paths.folder_names_are_relative_directory_paths_too = False
                                do_add(model_paths.additional_relative_directory_paths, index, path)
                                model_paths.folder_name_base_path_subdir = construct_path()
                            else:
                                do_add(model_paths.folder_names, index, potential_folder_name)
                except ValueError:
                    # this means that the relative directory didn't contain the subdir so far
                    # something_not_models/key
                    if potential_folder_name == key:
                        model_paths.folder_name_base_path_subdir = path.parent
                        model_paths.folder_names_are_relative_directory_paths_too = True
                    else:
                        if any(candidate.has_folder_name(potential_folder_name) for candidate in self.contents):
                            model_paths.folder_names_are_relative_directory_paths_too = False
                            do_add(model_paths.additional_relative_directory_paths, index, path)
                            model_paths.folder_name_base_path_subdir = construct_path()
                        else:
                            do_add(model_paths.folder_names, index, potential_folder_name)
        return model_paths

    def __setitem__(self, key: str, value: tuple | FolderPathsTuple | AbstractPaths):
        # remove all existing paths for this key
        self.__delitem__(key)

        if isinstance(value, AbstractPaths):
            self.contents.append(value)
        elif isinstance(value, (tuple, FolderPathsTuple)):
            paths, supported_extensions = value
            # typical cases:
            # key="checkpoints", paths="C:/base_path/models/checkpoints"
            # key="unets", paths="C:/base_path/models/unets", "C:/base_path/models/diffusion_models"
            #   ^ in this case, we will want folder_names to be both unets and diffusion_models
            # key="custom_loader", paths="C:/base_path/models/checkpoints"

            # if the paths are subdirectories of any basepath, use relative paths
            paths: list[Path] = list(map(Path, paths))
            self.contents.append(self._modify_model_paths(key, paths, supported_extensions))

    def __len__(self):
        return len(self.contents)

    def __iter__(self) -> typing.Generator[str]:
        for model_paths in self.contents:
            for folder_name in model_paths.folder_names:
                yield folder_name

    def __delitem__(self, key):
        to_remove: list[AbstractPaths] = []
        if isinstance(key, str):
            folder_names = [key]
        else:
            iter(key)
            folder_names = key

        for model_paths in self.contents:
            for folder_name in folder_names:
                if model_paths.has_folder_name(folder_name):
                    to_remove.append(model_paths)

        for model_paths in to_remove:
            self.contents.remove(model_paths)

    def __contains__(self, item: str):
        return any(candidate.has_folder_name(item) for candidate in self.contents)

    def items(self):
        items_view = {
            folder_name: self[folder_name] for folder_name in self.keys()
        }
        return items_view.items()

    def values(self):
        return [self[folder_name] for folder_name in self.keys()]

    def keys(self):
        return [x for x in self]

    def get(self, key, __default=None):
        for candidate in self.contents:
            if candidate.has_folder_name(key):
                return FolderPathsTuple(key, parent=weakref.ref(self))
        if __default is not None:
            raise ValueError("get with default is not supported")

    def copy(self):
        return copy.deepcopy(self)

    def clear(self):
        if self.is_root:
            logger.warning(f"trying to clear the root folder names and paths instance, this will cause unexpected behavior")
        self.contents = []


class SaveImagePathTuple(NamedTuple):
    full_output_folder: str
    filename: str
    counter: int
    subfolder: str
    filename_prefix: str
