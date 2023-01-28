import os
import site

def set_path_via_site():
    root_path = os.path.join(os.getcwd())
    src_path = os.path.join(os.getcwd(), "src")
    FIND_MY_PACKAGES = (
        f'import site\nsite.addsitedir("{root_path}")\nsite.addsitedir("{src_path}")'.replace('\\','\\\\')
    )
    try:
        os.makedirs(site.USER_SITE)
    except:
        pass
    filename = os.path.join(site.USER_SITE, "sitecustomize.py")
    with open(filename, "w") as outfile:
        print(FIND_MY_PACKAGES, file=outfile)
        print(f'Path set to "{root_path}" and "{src_path}"')

def handle_folders_structure():
    create_folder("checkpoints")
    create_folder("assets")
    create_folder("logs")
    create_folder("temp")
    create_folder("results")

def create_folder(name):
    try:
        os.mkdir(name)
    except:
        pass


def main():
    set_path_via_site()
    handle_folders_structure()


if __name__ == "__main__":
    main()
