import sys
import os.path
import pprint
import argparse
import ruamel.yaml
import folder_paths

yaml = ruamel.yaml.YAML()
yaml.default_flow_style = False
yaml.sort_keys = False
CM = ruamel.yaml.comments.CommentedMap

class AbstractOptionInfo:
    """
    A single option that can be saved to the config file YAML. Can potentially
    comprise more than one command line arg/flag like mutually exclusive flag
    groups being condensed into an enum in the config file
    """

    name: str
    raw_output: bool

    def add_argument(self, parser, suppress):
        """
        Adds an argument to the argparse parser
        """
        pass

    def get_arg_defaults(self, parser):
        """
        Returns the expected argparse namespaced options as a dictionary from
        querying the parser's default value
        """
        pass

    def get_help(self):
        pass

    def convert_to_args_array(self, value):
        """
        Interprets this option and a value as a string array of argstrings

        If it's a flag returns ["--flag"] for True and [] for False, otherwise
        can return ["--option", str(value)] or similar

        Alternatively if raw_output is True the parser will skip the parsing
        step and use the value as if it were returned from parse_known_args()
        """
        pass

    def convert_to_file_option(self, parser, args):
        """
        Converts a portion of the args to a value to be serialized to the config
        file

        As an example vram options are mutually exclusive, so it's made to look
        like an enum in the config file. So having a --lowvram flag in the args
        gets translated to "vram: 'lowvram'" in YAML
        """
        pass

    def validate(self, config_options, cli_args):
        """
        Modifies config_options to fix inconsistencies

        Example: config sets an enum value, cli args sets another, the flags
        should be removed from config_options
        """
        pass

class OptionInfo(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args
        self.raw_output = False

    def __repr__(self):
        return f'OptionInfo(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser, suppress=False):
        parser_args = dict(self.parser_args)
        if suppress:
            parser_args["default"] = argparse.SUPPRESS
        parser.add_argument(f"--{self.name}", **parser_args)

    def get_arg_defaults(self, parser):
        return { self.name: parser.get_default(self.name) }

    def get_help(self):
        help = self.parser_args.get("help")
        if help is None:
            return None
        type = self.parser_args.get("type")
        if type:
            help += f"\nType: {type.__name__}"
        return help

    def convert_to_args_array(self, value):
        if value is not None:
            return [f"--{self.name}", str(value)]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), parser.get_default(self.name))

    def validate(self, config_options, cli_args):
        pass

class OptionInfoFlag(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args
        self.raw_output = False

    def __repr__(self):
        return f'OptionInfoFlag(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser, suppress):
        parser_args = dict(self.parser_args)
        if suppress:
            parser_args["default"] = argparse.SUPPRESS
        parser.add_argument(f"--{self.name}", action="store_true", **parser_args)

    def get_arg_defaults(self, parser):
        return { self.name.replace("-", "_"): parser.get_default(self.name) or False }

    def get_help(self):
        help = self.parser_args.get("help")
        if help is None:
            return None
        help += "\nType: bool"
        return help

    def convert_to_args_array(self, value):
        if value:
            return [f"--{self.name}"]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), parser.get_default(self.name) or False)

    def validate(self, config_options, cli_args):
        pass

class OptionInfoEnum(AbstractOptionInfo):
    def __init__(self, name, options, help=None, empty_help=None):
        self.name = name
        self.options = options
        self.help = help
        self.empty_help = empty_help
        self.parser_args = {}
        self.raw_output = True

    def __repr__(self):
        return f'OptionInfoEnum(\'{self.name}\', {pprint.pformat(self.options)})'

    def add_argument(self, parser, suppress):
        group = parser.add_mutually_exclusive_group()
        default = None
        if suppress:
            default = argparse.SUPPRESS
        for option in self.options:
            group.add_argument(f"--{option.option_name}", action="store_true", help=option.help, default=default)

    def get_arg_defaults(self, parser):
        result = {}
        for option in self.options:
            result[option.option_name.replace("-", "_")] = False
        return result

    def get_help(self):
        if self.help is None:
            return None
        help = self.help + "\nChoices:"

        help += "\n - (empty)"
        if self.empty_help is not None:
            help += f": {self.empty_help}"

        for option in self.options:
            help += f"\n - {option.name}"
            if option.help:
                help += f": {option.help}"

        return help

    def convert_to_args_array(self, file_value):
        affected_options = [o.option_name.replace("-", "_") for o in self.options]
        for option in self.options:
            if option.name == file_value:
                return ({ option.option_name: True }, affected_options)
        return ({}, affected_options)

    def convert_to_file_option(self, parser, args):
        for option in self.options:
            if args.get(option.option_name.replace("-", "_")) is True:
                return option.name
        return None

    def validate(self, config_options, cli_args):
        set_by_cli = any(o for o in self.options if cli_args.get(o.option_name.replace("-", "_")) is not None)
        if set_by_cli:
            for option in self.options:
                config_options[option.option_name.replace("-", "_")] = False

class OptionInfoEnumChoice:
    name: str
    option_name: str
    help: str

    def __init__(self, name, option_name=None, help=None):
        self.name = name
        if option_name is None:
            option_name = self.name
        self.option_name = option_name
        if help is None:
            help = ""
        self.help = help

    def __repr__(self):
        return f'OptionInfoEnumChoice(\'{self.name}\', \'{self.option_name}\', \'{self.help}\')'

class OptionInfoRaw:
    """
    Raw YAML input and output, ignores argparse entirely
    """

    def __init__(self, name, help=None):
        self.name = name
        self.help = help
        self.parser_args = {}
        self.raw_output = True

    def add_argument(self, parser, suppress):
        pass

    def get_help(self):
        return self.help

    def get_arg_defaults(self, parser):
        return { self.name: {} }

    def convert_to_args_array(self, value):
        return { self.name: value }

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), {})

    def validate(self, config_options, cli_args):
        pass

#
# Config options
#

CONFIG_OPTIONS = [
    ("network", [
        OptionInfo("listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0",
                   help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)"),
        OptionInfo("port", type=int, default=8188, help="Set the listen port."),
        OptionInfo("enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                   help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'."),
    ]),
    ("files", [
        OptionInfoRaw("extra-model-paths", help="Extra paths to scan for model files."),
        OptionInfo("output-directory", type=str, default=None, help="Set the ComfyUI output directory. Leave empty to use the default."),
    ]),
    ("behavior", [
        OptionInfoFlag("auto-launch",
                       help="Automatically launch ComfyUI in the default browser."),
        OptionInfoFlag("dont-print-server",
                       help="Don't print server output."),
        OptionInfoFlag("quick-test-for-ci",
                       help="Quick test for CI."),
        OptionInfoFlag("windows-standalone-build",
                       help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup)."),
    ]),
    ("pytorch", [
        OptionInfo("cuda-device", type=int, default=None, metavar="DEVICE_ID",
                   help="Set the id of the cuda device this instance will use, or leave empty to autodetect."),
        OptionInfoFlag("dont-upcast-attention",
                       help="Disable upcasting of attention. Can boost speed but increase the chances of black images."),
        OptionInfoFlag("force-fp32",
                   help="Force fp32 (If this makes your GPU work better please report it)."),
        OptionInfo("directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1,
                   help="Use torch-directml."),
        OptionInfoEnum("cross-attention", [
            OptionInfoEnumChoice("split", option_name="use-split-cross-attention", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory."),
            OptionInfoEnumChoice("pytorch", option_name="use-pytorch-cross-attention", help="Used to force normal vram use if lowvram gets automatically enabled."),
        ], help="Type of cross attention to use", empty_help="Don't use cross-attention."),
        OptionInfoFlag("disable-xformers",
                       help="Disable xformers."),
        OptionInfoEnum("vram", [
            OptionInfoEnumChoice("highvram", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory."),
            OptionInfoEnumChoice("normalvram", help="Used to force normal vram use if lowvram gets automatically enabled."),
            OptionInfoEnumChoice("lowvram", help="Split the unet in parts to use less vram."),
            OptionInfoEnumChoice("novram", help="When lowvram isn't enough."),
            OptionInfoEnumChoice("cpu", help="To use the CPU for everything (slow).")
        ], help="Determines how VRAM is used.", empty_help="Autodetect the optional VRAM settings based on hardware.")
    ])
]

#
# Config parser
#

def make_config_parser(option_infos, suppress=False):
    parser = argparse.ArgumentParser()

    for category, options in option_infos:
        for option in options:
            option.add_argument(parser, suppress)

    return parser

def merge_dicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(merge_dicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

def recursive_delete_comment_attribs(d):
    if isinstance(d, dict):
        for k, v in d.items():
            recursive_delete_comment_attribs(k)
            recursive_delete_comment_attribs(v)
    elif isinstance(d, list):
        for elem in d:
            recursive_delete_comment_attribs(elem)
    try:
         # literal scalarstring might have comment associated with them
         attr = 'comment' if isinstance(d, ruamel.yaml.scalarstring.ScalarString) \
                  else ruamel.yaml.comments.Comment.attrib
         delattr(d, attr)
    except AttributeError:
        pass

class ComfyConfigLoader:
    def __init__(self):
        self.option_infos = CONFIG_OPTIONS
        self.parser = make_config_parser(self.option_infos)

    def get_arg_defaults(self):
        defaults = {}

        for category, options in self.option_infos:
            for option in options:
                arg_defaults = option.get_arg_defaults(self.parser)
                for k, v in arg_defaults.items():
                    k = k.replace('-', '_')
                    defaults[k] = v

        return defaults

    def load_from_string(self, raw_config):
        raw_config = yaml.load(raw_config)

        config = {}
        root = raw_config.get("config", {})

        for category, options in self.option_infos:
            if category in root:
                from_file = root[category]

                known_args = []
                for k, v in from_file.items():
                    kebab_k = k.replace("_", "-")
                    option_info = next((o for o in options if o.name == kebab_k), None)
                    if option_info is not None:
                        known_args = option_info.convert_to_args_array(v)
                        affected_options = [k]
                        if isinstance(known_args, tuple):
                            # Enum options can affect more than one flag in the
                            # CLI args, so have to check multiple items in the
                            # namespace argparse returns
                            affected_options = known_args[1]
                            known_args = known_args[0]

                        if option_info.raw_output:
                            converted = {}
                            for k, v in known_args.items():
                                converted[k.replace("-", "_")] = v
                            parsed = argparse.Namespace(**converted)
                            rest = None
                        else:
                            parsed, rest = self.parser.parse_known_args(known_args)

                        parsed_vars = vars(parsed)

                        # parse_known_args returns *all* options configured even
                        # if they're not found in the argstring. So have to pick
                        # out only the args affected by this option.
                        for ka in affected_options:
                            underscore_ka = ka.replace("-", "_")
                            item = parsed_vars.get(underscore_ka)
                            if item is not None:
                                config[ka] = item

                        if rest:
                            print(f"Warning: unparsed args - {pprint.pformat(rest)}")

        return config

    def convert_args_to_options(self, args):
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # strip previous YAML comments
        recursive_delete_comment_attribs(args)

        config = {}
        for category, options in self.option_infos:
            d = CM()
            first = True
            for option in options:
                k = option.name.replace('-', '_')
                d[k] = option.convert_to_file_option(self.parser, args)
                help = option.get_help()
                if help is not None:
                    help_string = "\n" + help
                    d.yaml_set_comment_before_after_key(k, help_string, indent=4)
                first = False
            config[category] = d
        return { "config": config }

    def save_config(self, config_path, args):
        options = self.convert_args_to_options(args)
        with open(config_path, 'w') as f:
            yaml.dump(options, f)

    def get_cli_arguments(self, argv):
        # first parse regularly and exit if an error is found
        self.parser.parse_args(argv)

        # now create another parser that suppresses missing arguments (not
        # user-specified) such that only the arguments passed will be put in the
        # namespace. Without this every argument set in the config will be
        # overridden because they're all present in the argparse.Namespace
        suppressed_parser = make_config_parser(self.option_infos, suppress=True)
        return vars(suppressed_parser.parse_args(argv))

    def parse_args_with_file(self, yaml_path, argv):
        if not os.path.isfile(yaml_path):
            print(f"Warning: no config file at path '{yaml_path}', creating it")
            raw_config = "{}"
        else:
            with open(yaml_path, 'r') as stream:
                raw_config = stream.read()
        return self.parse_args_with_string(raw_config, argv, save_config_file=yaml_path)

    def parse_args_with_string(self, config_string, argv, save_config_file=None):
        defaults = self.get_arg_defaults()

        config_options = self.load_from_string(config_string)
        config_options = dict(merge_dicts(defaults, config_options))
        if save_config_file:
            self.save_config(save_config_file, config_options)

        cli_args = self.get_cli_arguments(argv)

        for category, options in self.option_infos:
            for option in options:
                option.validate(config_options, cli_args)

        args = dict(merge_dicts(config_options, cli_args))
        return argparse.Namespace(**args)

args = {}

if "pytest" not in sys.modules:
    #
    # Load config and CLI args
    #

    config_loader = ComfyConfigLoader()
    args = config_loader.parse_args_with_file(folder_paths.default_config_path, sys.argv[1:])

    if args.windows_standalone_build:
        args.auto_launch = True
