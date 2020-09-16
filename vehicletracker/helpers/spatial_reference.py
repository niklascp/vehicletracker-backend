from typing import List

class SpatialRef():
    def __init__(self):
        self._line_refs = None
        self._stop_point_refs = None
        self._link_refs = None

    @property    
    def line_refs(self) -> List[str]:
        return self._line_refs

    @property    
    def stop_point_refs(self) -> List[str]:
        return self._stop_point_refs

    @property    
    def link_refs(self) -> List[str]:
        return self._link_refs

    def __repr__(self):
        str_ = ''
        if (self._line_refs):
            str_ += 'L:' + self._line_refs[0] + '+'
        if (self._stop_point_refs):
            str_ += 'SP:' + self._stop_point_refs[0] + '+'
        if (self._link_refs):
            str_ += 'LN:' + self._link_refs[0] + '+'
        return str_[:-1]

def parse_spatial_ref(str_repr: str) -> SpatialRef:
    spatial_ref = SpatialRef()

    if str_repr.startswith('L:'):
        spatial_ref._line_refs = [str_repr.split(':', 2)[1]]
    elif str_repr.startswith('SP:'):
        spatial_ref._stop_point_refs = [str_repr.split(':', 2)[1]]
    elif str_repr.startswith('LN:'):
        spatial_ref._link_refs = [str_repr.split(':', 2)[1]]

    return spatial_ref
