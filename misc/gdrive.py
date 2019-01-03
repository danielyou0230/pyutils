import argparse
import re
import subprocess
"""
python gdrive.py https://drive.google.com/file/d/1dj3vc8ZN82kmb7X7Hh9YtLem0-LwCAXU/view xx.pdf
"""
def from_gd(link, output="output"):
    # Pattern and link template
    pattern = "https:\/\/drive\.google\.com\/file\/d\/(.*)\/view"
    directdl_prefix = "https://drive.google.com/uc?export=download\&id="

    print("Got url: {}".format(link))
    file_id = re.findall(pattern, link)
    assert len(file_id) > 0
    print("Found id: {}".format(file_id))

    cmd = "wget {}{} -O {}".format(directdl_prefix, file_id[0], output)
    cmd = "wget {}{}".format(directdl_prefix, file_id[0])
    print(cmd)
    subprocess.run(cmd, shell=True)

def main(args):
    from_gd(args.url, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Link to the Google Drive file.")
    parser.add_argument("output", type=str, help="Output filename.")
    args = parser.parse_args()

    main(args)