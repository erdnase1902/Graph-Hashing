from utils import exec_cmd
import subprocess
from collections import OrderedDict
from os.path import basename, isfile, join

proj = 'OurMCS'
# proj = 'OurBiGNN'
server = 'yba@qilin.cs.ucla.edu'
remote_path = '/home/yba/GraphMatching/model/{}/logs'.format(proj)
local_path = '/home/yba/Documents/GraphMatching/model/{}/logs'.format(proj)
end_markers = ['final_test_pairs.klepto', 'exception.txt']


# ['gen_type=BA_2020-04-19T15-01-31.020629']

def main():
    local_fs = get_fs_dict(local_path, '')
    remote_cmd = 'ssh {}'.format(server)
    remote_fs = get_fs_dict(remote_path, remote_cmd)
    print('Found {} remote fs and {} local fs'.
          format(len(remote_fs), len(local_fs)))
    fps = gen_files_to_fetch(local_fs, remote_fs)
    print('{} fs to fetch'.format(len(fps)))
    commands = []
    ay = prompt('Want ALL {} files? [y/n]'.format(len(fps)), ['y', 'n'])
    for fp in fps:
        fbase = basename(fp)
        if ay == 'y':
            a = 'y'
        else:
            a = prompt('Want {}? [y/n]'.format(fbase), ['y', 'n'])
        if a == 'y':
            scp_cmd = 'scp -r "{}:{}" {}'.format(server, fp.replace('|', '\|').replace(' ', '\ ').replace("'", "\\'"), local_path)
            commands.append(scp_cmd)
            print(scp_cmd)
    for cmd in commands:
        exec_cmd(cmd)


def get_fs_dict(folder, remote_cmd):
    if not folder.endswith('/*'):
        folder += '/*'
    if remote_cmd != '' and not remote_cmd.endswith(' '):
        remote_cmd += ' '
    cmd = '{}du -sh {}'.format(remote_cmd, folder)
    print(cmd)
    # cmd = 'cd / && ls -l'
    out = subprocess.Popen(cmd.split(),
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           shell=remote_cmd == '',
                           cwd=local_path)
    stdout, stderr = out.communicate()
    if stderr is not None:
        raise RuntimeError(stderr)

    stdout = str(stdout)
    assert stdout[0:2] == 'b"' and stdout[-3:] == '\\n"', \
        '{} and {}'.format(stdout[0:2], stdout[-3:])
    stdout = stdout[2:-3]
    list_of_fs = []

    for f in stdout.split('\\n'):
        li = f.split('\\t')
        if len(li) != 2:
            raise RuntimeError('Wrong: {}'.format(stdout))
        assert len(li) == 2, li
        size, fp = li[0], li[1]
        if remote_cmd == '':  # local
            fp = join(local_path, fp)
        list_of_fs.append((size, fp))

    list_of_fs.sort(key=lambda x: x[1])

    fs_dict = OrderedDict()
    for size, fp in list_of_fs:
        fs_dict[fp] = size

    assert len(list_of_fs) == len(fs_dict)

    return fs_dict


def gen_files_to_fetch(local_fs, remote_fs):
    rtn = []
    for fp, size in remote_fs.items():
        if not find_file(fp, size, local_fs):
            rtn.append(fp)
    return rtn


def find_file(remote_fp, remote_size, local_fs):
    for local_fp, local_size in local_fs.items():
        # x = basename(remote_fp) == basename(local_fp)
        if (basename(remote_fp) == basename(local_fp) or
            basename(remote_fp) in basename(local_fp)) \
                and has_end_marker(
            local_fp):  # local folder already contains ending file TODO: may think not finished if no marker file generated on server in the first place
            return True
    return False


def has_end_marker(local_fp):
    for em in end_markers:
        if isfile(join(local_fp, em)):
            return True
    return False


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


if __name__ == '__main__':
    main()
