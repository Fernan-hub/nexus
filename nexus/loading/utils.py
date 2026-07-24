"""Helpers for flattening Neo block/segment hierarchies into flat data lists."""

from neo.core import Block, Segment
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy


def _update_with_block_and_segment_metadata(
    data_list: list[BaseProxy] | list[DataObject],
    block: Block,
    segment: Segment,
) -> None:
    """Annotate each object with its parent block and segment names and annotations.

    Parameters
    ----------
    data_list : list[BaseProxy] | list[DataObject]
        Objects to annotate in-place.
    block : Block
        Parent block whose name and annotations are merged in.
    segment : Segment
        Parent segment whose name and annotations are merged in.
    """
    for data in data_list:
        data.annotate(
            block_name=block.name,
            segment_name=segment.name,
            **block.annotations,
            **segment.annotations,
        )


def get_all_data_from_single_block(block: Block) -> list[BaseProxy] | list[DataObject]:
    """Return all analog and irregularly sampled signals from every segment.

    Parameters
    ----------
    block : Block
        Neo block from which signals are extracted.

    Returns
    -------
    list[BaseProxy] | list[DataObject]
        Flat list of all signals across all segments, annotated with block and
        segment metadata.
    """
    data_list: list[BaseProxy] | list[DataObject] = []
    for seg in block.segments:
        segment_data = seg.analogsignals + seg.irregularlysampledsignals
        _update_with_block_and_segment_metadata(segment_data, block, seg)
        data_list.extend(segment_data)
    return data_list


def get_all_data_from_blocks(blocks: list[Block]) -> list[BaseProxy] | list[DataObject]:
    """Return all signals from every block in a flat list.

    Parameters
    ----------
    blocks : list[Block]
        Neo blocks from which signals are extracted.

    Returns
    -------
    list[BaseProxy] | list[DataObject]
        Flat list of all signals across all blocks and segments.
    """
    data_list: list[BaseProxy] | list[DataObject] = []
    for block in blocks:
        data_list.extend(get_all_data_from_single_block(block))
    return data_list
