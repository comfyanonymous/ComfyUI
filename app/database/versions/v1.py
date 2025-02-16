from folder_paths import folder_names_and_paths, get_filename_list, get_full_path


def v1(cursor):
    print("Updating to v1")
    for folder_name in folder_names_and_paths.keys():
        if folder_name == "custom_nodes":
            continue

        files = get_filename_list(folder_name)
        for file in files:
            file_path = get_full_path(folder_name, file)
            file_without_extension = file.rsplit(".", maxsplit=1)[0]
            cursor.execute(
                "INSERT INTO models (path, name, type) VALUES (?, ?, ?)",
                (file_path, file_without_extension, folder_name),
            )
