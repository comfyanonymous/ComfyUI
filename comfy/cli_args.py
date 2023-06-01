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

    def add_argument(self, parser):
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

class OptionInfo(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args
        self.raw_output = False

    def __repr__(self):
        return f'OptionInfo(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser):
        parser.add_argument(f"--{self.name}", **self.parser_args)

    def get_arg_defaults(self, parser):
        return { self.name: parser.get_default(self.name) }

    def get_help(self):
        return self.parser_args.get("help")

    def convert_to_args_array(self, value):
        if value is not None:
            return [f"--{self.name}", str(value)]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), parser.get_default(self.name))

class OptionInfoFlag(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args
        self.raw_output = False

    def __repr__(self):
        return f'OptionInfoFlag(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser):
        parser.add_argument(f"--{self.name}", action="store_true", **self.parser_args)

    def get_arg_defaults(self, parser):
        return { self.name: parser.get_default(self.name) or False }

    def get_help(self):
        return self.parser_args.get("help")

    def convert_to_args_array(self, value):
        if value:
            return [f"--{self.name}"]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), parser.get_default(self.name) or False)

class OptionInfoEnum(AbstractOptionInfo):
    def __init__(self, name, options, help=None):
        self.name = name
        self.options = options
        self.help = help
        self.parser_args = {}
        self.raw_output = False

    def __repr__(self):
        return f'OptionInfoEnum(\'{self.name}\', {pprint.pformat(self.options)})'

    def add_argument(self, parser):
        group = parser.add_mutually_exclusive_group()
        for option in self.options:
            group.add_argument(f"--{option.option_name}", action="store_true", help=option.help)

    def get_arg_defaults(self, parser):
        return {} # treat as no flag in the group being passed

    def get_help(self):
        return self.help

    def convert_to_args_array(self, file_value):
        for option in self.options:
            if option.name == file_value:
                return [f"--{option.option_name}"]
        return []

    def convert_to_file_option(self, parser, args):
        for option in self.options:
            if args.get(option.option_name.replace("-", "_")) is True:
                return option.name
        return None

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

    def add_argument(self, parser):
        pass

    def get_help(self):
        return self.help

    def get_arg_defaults(self, parser):
        return { self.name: {} }

    def convert_to_args_array(self, value):
        return value

    def convert_to_file_option(self, parser, args):
        return args.get(self.name.replace("-", "_"), {})


CONFIG_OPTIONS = [
    ("network", [
        OptionInfo("listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0",
                   help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)"),
        OptionInfo("port", type=int, default=8188, help="Set the listen port."),
        OptionInfo("enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                   help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'."),
    ]),
    ("files", [
        OptionInfoRaw("extra-model-paths-config", help="Extra paths to scan for model files."),
        OptionInfo("output-directory", type=str, default=None, help="Set the ComfyUI output directory."),
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
                   help="Set the id of the cuda device this instance will use."),
        OptionInfoFlag("dont-upcast-attention",
                       help="Disable upcasting of attention. Can boost speed but increase the chances of black images."),
        OptionInfoFlag("force-fp32",
                   help="Force fp32 (If this makes your GPU work better please report it)."),
        OptionInfo("directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1,
                   help="Use torch-directml."),
        OptionInfoEnum("cross-attention", [
            OptionInfoEnumChoice("split", option_name="use-split-cross-attention", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory."),
            OptionInfoEnumChoice("pytorch", option_name="use-pytorch-cross-attention", help="Used to force normal vram use if lowvram gets automatically enabled."),
        ], help="Type of cross attention to use"),
        OptionInfoFlag("disable-xformers",
                       help="Disable xformers."),
        OptionInfoEnum("vram", [
            OptionInfoEnumChoice("highvram", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory."),
            OptionInfoEnumChoice("normalvram", help="Used to force normal vram use if lowvram gets automatically enabled."),
            OptionInfoEnumChoice("lowvram", help="Split the unet in parts to use less vram."),
            OptionInfoEnumChoice("novram", help="When lowvram isn't enough."),
            OptionInfoEnumChoice("cpu", help="To use the CPU for everything (slow).")
        ], help="Determines how VRAM is used.")
    ])
]


def make_config_parser(option_infos):
    parser = argparse.ArgumentParser()

    for category, options in option_infos:
        for option in options:
            option.add_argument(parser)

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

class ComfyConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
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

    def load_from_file(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            raw_config = yaml.load(stream)

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
                        if option_info.raw_output:
                            parsed = argparse.Namespace(**{ k: known_args })
                            rest = None
                        else:
                            parsed, rest = self.parser.parse_known_args(known_args)
                        print("---------------------")
                        print(option_info.name)
                        print(known_args)
                        item = vars(parsed).get(k)
                        if item is not None:
                            config[k] = v

                        if rest:
                            print(f"Warning: unparsed args - {pprint.pformat(rest)}")

        return config

    def convert_args_to_options(self, args):
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        config = {}
        for category, options in self.option_infos:
            d = CM()
            for option in options:
                k = option.name.replace('-', '_')
                d[k] = option.convert_to_file_option(self.parser, args)
                help = option.get_help()
                if help is not None:
                    d.yaml_set_comment_before_after_key(k, "\n" + help, indent=4)
            config[category] = d
        return { "config": config }

    def save_config(self, args):
        options = self.convert_args_to_options(args)
        with open(self.config_path, 'w') as f:
            yaml.dump(options, f)

    def get_cli_arguments(self):
        return vars(self.parser.parse_args())

    def parse_args(self):
        defaults = self.get_arg_defaults()

        if not os.path.isfile(self.config_path):
            print(f"Warning: no config file at path '{self.config_path}', creating it")
            config_options = {}
        else:
            config_options = self.load_from_file(self.config_path)

        config_options = dict(merge_dicts(defaults, config_options))
        self.save_config(config_options)

        cli_args = self.get_cli_arguments()
        print(cli_args)

        args = dict(merge_dicts(config_options, cli_args))
        return argparse.Namespace(**args)


config_loader = ComfyConfigLoader(folder_paths.default_config_path)
args = config_loader.parse_args()

if args.windows_standalone_build:
    args.auto_launch = True


import pprint; pprint.pp(args)
import pprint; pprint.pp(config_loader.convert_args_to_options(args))

exit(1)
