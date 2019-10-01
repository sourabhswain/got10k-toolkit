
class DAVIS:
    def __init__(self, root_dir, version="2017"):
        assert version == "2017", "DAVIS 2016 not implemented yet"
        self.seq_names = [None] # TODO

    def __getitem__(self, index):
        r"""
        Args:
          index (integer or string): Index or name of a sequence.

        Returns:
          tuple: (img_files, anno), where ``img_files`` is a list of
              file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        pass

    def __len__(self):
        return len(self.seq_names)
