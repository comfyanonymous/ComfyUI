import os.path
import pprint
import argparse
import yaml
import folder_paths

class AbstractOptionInfo:
    name: str

    def add_argument(self, parser):
        pass

    def get_arg_defaults(self, parser):
        pass

    def convert_to_args_array(self, value):
        pass

    def convert_to_file_option(self, parser, args):
        pass

    def save(self, yaml):
        pass

    def load(self, yaml):
        pass

class OptionInfo(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args

    def __repr__(self):
        return f'OptionInfo(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser):
        parser.add_argument(f"--{self.name}", **self.parser_args)

    def get_arg_defaults(self, parser):
        return { self.name: parser.get_default(self.name) }

    def convert_to_args_array(self, value):
        if value is not None:
            return [f"--{self.name}", str(value)]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name, parser.get_default(self.name))

    def save(self, yaml):
        pass

    def load(self, yaml):
        pass

class OptionInfoFlag(AbstractOptionInfo):
    def __init__(self, name, **parser_args):
        self.name = name
        self.parser_args = parser_args

    def __repr__(self):
        return f'OptionInfoFlag(\'{self.name}\', {pprint.pformat(self.parser_args)})'

    def add_argument(self, parser):
        parser.add_argument(f"--{self.name}", action="store_true", **self.parser_args)

    def get_arg_defaults(self, parser):
        return { self.name: parser.get_default(self.name) or False }

    def convert_to_args_array(self, value):
        if value:
            return [f"--{self.name}"]
        return []

    def convert_to_file_option(self, parser, args):
        return args.get(self.name, parser.get_default(self.name) or False)

    def save(self, yaml):
        pass

    def load(self, yaml):
        pass

class OptionInfoEnum(AbstractOptionInfo):
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def __repr__(self):
        return f'OptionInfoEnum(\'{self.name}\', {pprint.pformat(self.options)})'

    def add_argument(self, parser):
        group = parser.add_mutually_exclusive_group()
        for option in self.options:
            group.add_argument(f"--{option.option_name}", action="store_true", help=option.help)

    def get_arg_defaults(self, parser):
        return {} # treat as no flag in the group being passed

    def convert_to_args_array(self, file_value):
        for option in self.options:
            if option.name == file_value:
                return [f"--{option.option_name}"]
        return []

    def convert_to_file_option(self, parser, args):
        for option in self.options:
            if args.get(option.option_name) is True:
                return option.name
        return None

    def load_from_yaml(self, yaml):
        pass

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


CONFIG_OPTIONS = [
    ("network", [
        OptionInfo("listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0",
                   help="Specify the IP address to listen on (default: 127.0.0.1). \
                     If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)"),
        OptionInfo("port", type=int, default=8188, help="Set the listen port."),
        OptionInfo("enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                   help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'."),
    ]),
    ("files", [
        OptionInfo("extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append',
                   help="Load one or more extra_model_paths.yaml files."),
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
        ]),
        OptionInfoFlag("disable-xformers",
                       help="Disable xformers."),
        OptionInfoEnum("vram", [
            OptionInfoEnumChoice("highvram", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory."),
            OptionInfoEnumChoice("normalvram", help="Used to force normal vram use if lowvram gets automatically enabled."),
            OptionInfoEnumChoice("lowvram", help="Split the unet in parts to use less vram."),
            OptionInfoEnumChoice("novram", help="When lowvram isn't enough."),
            OptionInfoEnumChoice("cpu", help="To use the CPU for everything (slow).")
        ])
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
            raw_config = yaml.safe_load(stream)

        config = {}
        root = raw_config.get("config", {})

        for category, options in self.option_infos:
            if category in root:
                from_file = root[category]

                known_args = []
                for k, v in from_file.items():
                    option_info = next((o for o in options if o.name == k), None)
                    if option_info is not None:
                        known_args = option_info.convert_to_args_array(v)
                        parsed, rest = self.parser.parse_known_args(known_args)
                        for k, v in vars(parsed).items():
                            k = k.replace("-", "_")
                            config[k] = v

                        if rest:
                            print(f"Warning: unparsed args - {pprint.pformat(rest)}")

        return config

    def convert_args_to_options(self, args):
        config = {}
        import pprint; pprint.pp(args)
        for category, options in self.option_infos:
            d = {}
            for option in options:
                k = option.name.replace('-', '_')
                d[k] = option.convert_to_file_option(self.parser, args)
            config[category] = d
        return { "config": config }

    def save_config(self, args):
        options = self.convert_args_to_options(args)
        with open(self.config_path, 'w') as f:
            yaml.dump(options, f, default_flow_style=False, sort_keys=False)

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
        self.save_config(defaults)

        cli_args = self.get_cli_arguments()

        args = dict(merge_dicts(config_options, cli_args))
        return argparse.Namespace(**args)


config_loader = ComfyConfigLoader(folder_paths.default_config_path)
args = config_loader.parse_args()

if args.windows_standalone_build:
    args.auto_launch = True
