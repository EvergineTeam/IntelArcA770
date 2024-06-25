import os

env_path = os.environ["PATH"]

env_path_list = env_path.split(";")
env_path_list_no_duplicates = list(dict.fromkeys(env_path_list))

env_path_no_duplicates = ";".join(env_path_list_no_duplicates)

print(env_path_no_duplicates)
