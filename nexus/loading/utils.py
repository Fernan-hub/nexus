from neo.core import Block, Segment
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy


def _update_with_block_and_segment_metadata(
    data_list: list[BaseProxy] | list[DataObject],
    block: Block,
    segment: Segment,
) -> None:
    for data in data_list:
        data.annotate(
            block_name=block.name,
            segment_name=segment.name,
            **block.annotations,
            **segment.annotations,
        )


def get_all_data_from_single_block(block: Block) -> list[BaseProxy] | list[DataObject]:
    data_list: list[BaseProxy] | list[DataObject] = []
    for seg in block.segments:
        segment_data = seg.analogsignals + seg.irregularlysampledsignals
        _update_with_block_and_segment_metadata(segment_data, block, seg)
        data_list.extend(segment_data)
    return data_list


def get_all_data_from_blocks(blocks: list[Block]) -> list[BaseProxy] | list[DataObject]:
    data_list: list[BaseProxy] | list[DataObject] = []
    for block in blocks:
        data_list.extend(get_all_data_from_single_block(block))
    return data_list
