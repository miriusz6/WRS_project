from os import getcwd

class DefaultFrame():
    def __init__(self, loc_path, name, tag, description):
        self.path = loc_path
        self.name = name
        self.tag = tag
        self.description = description


# { (path, tag): (name, description) }
# frames: dict = {
#     ('df_metadata', 'General'): ('df_metadata','cleaned metadata (iid,title,description)'),
#     'df_tfidf'): ('df_tfidf','tfidf vector embedding for items(iid,tfidf_sum)', 'TFIDF'),

#     'sim_matrix_item_tfidf': ('sim_matrix_item_tfidf','similarity matrix for tfidf vectors of items', 'TFIDF'),
#     'General': '()',
# }

frames = [
    DefaultFrame(loc_path = 'df_metadata',
                 name = 'df_metadata', 
                 tag = 'General', 
                 description = 'cleaned metadata (iid,title,description)'),
    # TFIDF
        # description
    DefaultFrame(loc_path = 'df_tfidf_descr_item',
                    name = 'df_descr_item', 
                    tag = 'TFIDF', 
                    description = 'tfidf vector embedding for items(iid,tfidf_sum) based on description'),
    DefaultFrame(loc_path = 'sim_matrix_tfidf_descr_item',
                    name = 'sim_matrix_item', 
                    tag = 'TFIDF', 
                    description = 'similarity matrix for tfidf vectors of items based on description'),
        # title
    DefaultFrame(loc_path = 'df_tfidf_title_item',
                    name = 'df_descr_item', 
                    tag = 'TFIDF', 
                    description = 'tfidf vector embedding for items(iid,tfidf_sum)'),
    DefaultFrame(loc_path = 'sim_matrix_tfidf_item',
                    name = 'sim_matrix_item', 
                    tag = 'TFIDF', 
                    description = 'similarity matrix for tfidf vectors of items'),
    # W2V
    DefaultFrame(loc_path = 'df_w2v_descr_item',
                    name = 'df_descr_item', 
                    tag = 'W2V', 
                    description = 'w2v vector embedding for items(iid,tfidf_sum)'),
    DefaultFrame(loc_path = 'sim_matrix_w2v_item',
                    name = 'sim_matrix_item', 
                    tag = 'W2V', 
                    description = 'similarity matrix for w2v vectors of items'),
]


class FrameManager():
    # no tag nesting, no multiple tags per frame
    def __init__(self):
        self.frames = {}
        self.tags = {}
        self.add_tag('General', 'General-use frames')
        self.def_dir = getcwd()+"/saved_frames/"
        self.load_deafualt_from_file()


    def add_tag(self, tag, description):
        self.tags[tag] = description
    
    def add_frame(self, name, frame, description, tag: str = 'General'):
        if tag not in self.tags:
            raise ValueError(f"Tag {tag} is not defined")
        self.frames[(name, tag)] = (frame, description)

    def list_frames(self, tag: str = None):
        if tag is None:
            return str(self)
        else:
            s = ""
            for name,tag in self.frames.keys():
                if tag == tag:
                    meta = self.frames[(name, tag)]
                    s += f"{name}({tag}): {meta[1]}\n"
            return s

    # define print
    def __str__(self):
        s = "Tags:\n"
        for tag, description in self.tags.items():
            s += f"{tag}: {description}\n"
        s += "\nFrames:\n"
        for name,tag in self.frames.keys():
            meta = self.frames[(name, tag)]
            s += f"{name}({tag}): {meta[1]}\n"
        return s

    def load_deafualt_from_file(self):
        pass