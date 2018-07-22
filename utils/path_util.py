# coding: utf-8
# created by deng on 7/23/2018

from os.path import dirname, abspath, join, normpath

# 得到项目根目录
project_root_url = normpath(join(dirname(__file__), '..'))


def from_project_root(rel_path):
    """根据相对项目根目录的路径返回绝对路径

    Args:
        rel_path: 相对路径

    Returns:
        str: 绝对路径

    """
    return normpath(join(project_root_url, rel_path))


def main():
    print(project_root_url)
    print(from_project_root('.gitignore'))
    print(from_project_root('data/test.py'))
    pass


if __name__ == '__main__':
    main()
