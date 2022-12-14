from .deeppcb import Defect
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def showdefects(img, defects: List[Defect]):
    """
    show the given list of defects superimposed on top of an image.
    """

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img)
    for defect in defects:
        rect = patches.Rectangle((defect.x0,defect.y0), defect.width, defect.height,
                                facecolor='none', edgecolor='r')
        ax.add_patch(rect)
        ax.text(defect.x0, defect.y0-5, f'{defect.ty.name}', color='r')

    return fig, ax
