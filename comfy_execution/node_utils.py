import re
from typing import Optional, Tuple

# This decorator can be used to enable a "template" syntax for types in a node.
#
# Dynamic Types
# When specifying a type for an input or output, you can wrap an arbitrary string in angle brackets to indicate that it is dynamic. For example, the type "<FOO>" will be the equivalent of "*" (with the commonly used hacks) with the caveat that all inputs/outputs with the same template name ("FOO" in this case) must have the same type. Use multiple different template names if you want to allow types to differ. Note that this only applies within a single instance of a node -- different nodes can have different type resolutions
#
# Wrapping Types
# Rather than using JUST a template type, you can also use a template type with a wrapping type. For example, if you have a node that takes two inputs with the types "<FOO>" and "Accumulation<FOO>" respectively, any output can be connected to the "<FOO>" input. Once that input has a value (let's say an IMAGE), the other input will resolve as well (to Accumulation<IMAGE> in this example).
#
# Variadic Inputs
# Sometimes, you want a node to take a dynamic number of inputs. To do this, create an input value that has a name followed by a number sign and a string (e.g. "input#COUNT"). This will cause additional inputs to be added and removed as the user attaches to those sockets. The string after the '#' can be used to ensure that you have the same number of sockets for two different inputs. For example, having inputs named "image#FOO" and "mask#BAR" will allow the number of images and the number of masks to dynamically increase independently. Having inputs named "image#FOO" and "mask#FOO" will ensure that there are the same number of images as masks.
#
# Variadic Input - Same Type
# If you want to have a variadic input with a dynamic type, you can combine the two. For example, if you have an input named "input#COUNT" with the type "<FOO>", you can attach multiple inputs to that socket. Once you attach a value to one of the inputs, all of the other inputs will resolve to the same type. This is useful for nodes that take a dynamic number of inputs of the same type.
#
# Variadic Input - Different Types
# If you want to have a variadic input with a dynamic type, you can combine the two. For example, if you have an input named "input#COUNT" with the type "<FOO#COUNT>", each socket for the input can have a different type. (Internally, this is equivalent to making the type <FOO1> where 1 is the index of this input.)

def TemplateTypeSupport():
    def decorator(cls):
        old_input_types = getattr(cls, "INPUT_TYPES")
        def new_input_types(cls):
            old_types = old_input_types()
            new_types = {
                "required": {},
                "optional": {},
                "hidden": old_types.get("hidden", {}),
            }
            for category in ["required", "optional"]:
                if category not in old_types:
                    continue
                for key, value in old_types[category].items():
                    input_name = replace_variadic_suffix(key, 1)
                    input_type = template_to_type(value[0])
                    if len(value) == 1:
                        extra_info = {}
                    else:
                        extra_info = value[1]
                    if input_name != key or input_type != value[0]:
                        # TODO - Fix front-end to handle widgets and remove this
                        extra_info["forceInput"] = True
                    new_types[category][input_name] = (input_type,extra_info)
            return new_types
        setattr(cls, "INPUT_TYPES", classmethod(new_input_types))
        old_outputs = getattr(cls, "RETURN_TYPES")
        setattr(cls, "RETURN_TYPES", tuple(template_to_type(x) for x in old_outputs))

        def resolve_dynamic_types(cls, input_types, output_types, entangled_types):
            original_inputs = old_input_types()

            # Step 1 - Find all variadic groups and determine their maximum used index
            variadic_group_map = {}
            max_group_index = {}
            for category in ["required", "optional"]:
                for key, value in original_inputs.get(category, {}).items():
                    root, group = determine_variadic_group(key)
                    if root is not None and group is not None:
                        variadic_group_map[root] = group
            for type_map in [input_types, output_types]:
                for key in type_map.keys():
                    root, index = determine_variadic_suffix(key)
                    if root is not None and index is not None:
                        if root in variadic_group_map:
                            group = variadic_group_map[root]
                            max_group_index[group] = max(max_group_index.get(group, 0), index)

            # Step 2 - Create variadic arguments
            variadic_inputs = {
                "required": {},
                "optional": {},
            }
            for category in ["required", "optional"]:
                for key, value in original_inputs.get(category, {}).items():
                    root, group = determine_variadic_group(key)
                    if root is None or group is None:
                        # Copy it over as-is
                        variadic_inputs[category][key] = value
                    else:
                        for i in range(1, max_group_index.get(group, 0) + 2):
                            # Also replace any variadic suffixes in the type (for use with templates)
                            input_type = value[0]
                            if isinstance(input_type, str):
                                input_type = replace_variadic_suffix(input_type, i)
                            if len(value) == 1:
                                extra_info = {}
                            else:
                                extra_info = value[1]
                            if input_type != value[0]:
                                # TODO - Fix front-end to handle widgets and remove this
                                extra_info["forceInput"] = True
                            variadic_inputs[category][replace_variadic_suffix(key, i)] = (input_type,extra_info)

            # Step 3 - Resolve template arguments
            resolved = {}
            for category in ["required", "optional"]:
                for key, value in variadic_inputs[category].items():
                    if key in input_types:
                        tkey, tvalue = determine_template_value(value[0], input_types[key])
                        if tkey is not None and tvalue is not None:
                            resolved[tkey] = type_intersection(resolved.get(tkey, "*"), tvalue)
            for i in range(len(old_outputs)):
                output_name = cls.RETURN_NAMES[i]
                if output_name in output_types:
                    for output_type in output_types[output_name]:
                        tkey, tvalue = determine_template_value(old_outputs[i], output_type)
                        if tkey is not None and tvalue is not None:
                            resolved[tkey] = type_intersection(resolved.get(tkey, "*"), tvalue)

            # Step 4 - Replace templates with resolved types
            final_inputs = {
                "required": {},
                "optional": {},
                "hidden": original_inputs.get("hidden", {}),
            }
            for category in ["required", "optional"]:
                for key, value in variadic_inputs[category].items():
                    if len(value) == 1:
                        extra_info = {}
                    else:
                        extra_info = value[1]
                    resolved_type = template_to_type(value[0], resolved)
                    if resolved_type != value[0]:
                        # TODO - Fix front-end to handle widgets and remove this
                        extra_info["forceInput"] = True
                    final_inputs[category][key] = (resolved_type,extra_info)
            outputs = (template_to_type(x, resolved) for x in old_outputs)
            return {
                "input": final_inputs,
                "output": tuple(outputs),
                "output_name": cls.RETURN_NAMES,
                "dynamic_counts": max_group_index,
            }
        setattr(cls, "resolve_dynamic_types", classmethod(resolve_dynamic_types))
        return cls
    return decorator

def type_intersection(a: str, b: str) -> str:
    if a == "*":
        return b
    if b == "*":
        return a
    if a == b:
        return a
    aset = set(a.split(','))
    bset = set(b.split(','))
    intersection = aset.intersection(bset)
    if len(intersection) == 0:
        return "*"
    return ",".join(intersection)

naked_template_regex = re.compile(r"^<(.+)>$")
qualified_template_regex = re.compile(r"^(.+)<(.+)>$")
variadic_template_regex = re.compile(r"([^<]+)#([^>]+)")
variadic_suffix_regex =   re.compile(r"([^<]+)(\d+)")

empty_lookup = {}
def template_to_type(template, key_lookup=empty_lookup):
    templ_match = naked_template_regex.match(template)
    if templ_match:
        return key_lookup.get(templ_match.group(1), "*")
    templ_match = qualified_template_regex.match(template)
    if templ_match:
        resolved = key_lookup.get(templ_match.group(2), "*")
        return qualified_template_regex.sub(r"\1<%s>" % resolved, template)
    return template

# Returns the 'key' and 'value' of the template (if any)
def determine_template_value(template: str, actual_type: str) -> Tuple[Optional[str], Optional[str]]:
    templ_match = naked_template_regex.match(template)
    if templ_match:
        return templ_match.group(1), actual_type
    templ_match = qualified_template_regex.match(template)
    actual_match = qualified_template_regex.match(actual_type)
    if templ_match and actual_match and templ_match.group(1) == actual_match.group(1):
        return templ_match.group(2), actual_match.group(2)
    return None, None

def determine_variadic_group(template: str) -> Tuple[Optional[str], Optional[str]]:
    variadic_match = variadic_template_regex.match(template)
    if variadic_match:
        return variadic_match.group(1), variadic_match.group(2)
    return None, None

def replace_variadic_suffix(template: str, index: int) -> str:
    return variadic_template_regex.sub(lambda match: match.group(1) + str(index), template)

def determine_variadic_suffix(template: str) -> Tuple[Optional[str], Optional[int]]:
    variadic_match = variadic_suffix_regex.match(template)
    if variadic_match:
        return variadic_match.group(1), int(variadic_match.group(2))
    return None, None

