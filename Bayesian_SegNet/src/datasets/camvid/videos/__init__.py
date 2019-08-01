"""A python module to access raw video footage from CamVid."""
import os
import glob


def list_videos() -> list:
    """Return a list of the video files available."""
    # get a reference to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # create a glob for MP4 files in this directory
    mp4_glob = os.path.join(this_dir, '*.mp4')
    # return the output from the glob as a list
    return sorted(glob.glob(mp4_glob))


def abs_path(filename: str) -> str:
    """
    Return the absolute path to a video file in this directory.

    Args:
        filename: the name of a file in this directory to get the path of

    Returns:
        the absolute path to the given filename

    Raises:
        OSError: if the filename is not found in this package

    """
    # get a list of the videos in this directory
    videos = list_videos()
    # iterate over the videos
    for video in videos:
        if filename in video:
            return video
    raise OSError('{} is not a valid file in the videos package'.format(filename))


# explicitly define the outward facing API of this package
__all__ = [
    abs_path.__name__,
    list_videos.__name__,
]
