# Full Jass card list in fixed order (36 cards)
CARD_LIST = [
    "DA","DK","DQ","DJ","D10","D9","D8","D7","D6",
    "HA","HK","HQ","HJ","H10","H9","H8","H7","H6",
    "SA","SK","SQ","SJ","S10","S9","S8","S7","S6",
    "CA","CK","CQ","CJ","C10","C9","C8","C7","C6"
]

# Mapping card → index
CARD_TO_INDEX = {card: i for i, card in enumerate(CARD_LIST)}

# Mapping index → card
INDEX_TO_CARD = {i: card for i, card in enumerate(CARD_LIST)}
