"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    >>> bits_to_byte('00001110')
    14
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dict_ = {}
    for item in text:
        if item not in dict_:
            dict_[item] = 1
        else:
            dict_[item] += 1
    return dict_


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq = {1: 255}
    >>> t = huffman_tree(freq)
    >>> result = HuffmanNode(1)
    >>> t == result
    True
    """
    # helper 1
    freq_list_ = convert_dict(freq_dict)
    # helper 2
    return add_node_huffman(freq_list_)


# helper 1 : convert freq_dict to a sorted list
def convert_dict(freq_dict_):
    """ Return a sorted list of (freq, HuffmanNode) from the given freq_dict

    @param dict freq_dict_:
    @rtype: list

    >>> a = {253:1, 254: 2, 255: 3}
    >>> b = convert_dict(a)
    >>> b
    [(1, HuffmanNode(253, None, None)), (2, HuffmanNode(254, None, None)), \
(3, HuffmanNode(255, None, None))]
    """
    freq_list = []
    for symbol in freq_dict_:
        freq_list.append((freq_dict_[symbol], HuffmanNode(symbol)))
    freq_list.sort()
    return freq_list


# helper 2: generate a huffmantree
def add_node_huffman(list_bytes):
    """ Generate a huffman tree by sorted list of (freq, HuffmanNode)

    @param list list_bytes:
    @rtype: HuffmanNode

    >>> hftree = HuffmanNode()
    >>> list_bytes = [(1, HuffmanNode('a')), (3, HuffmanNode('b')), \
(10, HuffmanNode('c')), (12, HuffmanNode('d')), (15, HuffmanNode('e'))]
    >>> tree = add_node_huffman(list_bytes)
    >>> tree == HuffmanNode(None, HuffmanNode('e', None, None), \
HuffmanNode(None, HuffmanNode('d', None, None), HuffmanNode(None, \
HuffmanNode(None, HuffmanNode('a', None, None), \
HuffmanNode('b', None, None)), HuffmanNode('c', None, None))))
    True
    """
    while len(list_bytes) > 1:
        new_node = HuffmanNode(None, list_bytes[0][1], list_bytes[1][1])
        freq = list_bytes[0][0] + list_bytes[1][0]
        list_bytes.pop(0)
        list_bytes.pop(0)
        new_tuple = (freq, new_node)
        list_bytes.append(new_tuple)
        list_bytes.sort()
    return list_bytes[0][1]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    dict_code = {}
    # check if only one symbol
    if tree.is_leaf():
        dict_code[tree.symbol] = 0
    else:
        dict_code = add_code(tree, dict_code)
    return dict_code


# helper: accumulate code
def add_code(hfnode, dict_, code=''):
    """ helper for get_codes to add code recursively

    @param HuffmanNode hfnode:
    @param dict dict_:
    @param str code:

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if hfnode.is_leaf():
        dict_[hfnode.symbol] = code
    else:
        add_code(hfnode.left, dict_, code=code+'0')
        add_code(hfnode.right, dict_, code=code+'1')
        return dict_


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> tree.number
    0
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.number
    1
    >>> tree.left.number
    0
    """
    add_number(tree, 0)


# helper: add number recursively
def add_number(hfnode, num):
    """ helper for number_node to add number to internal nodes

    @param HuffmanNode hfnode:
    @param int num:
    @rtype: int

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> add_number(tree, 0)
    3
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if hfnode is not None:
        if not hfnode.is_leaf():
            num = add_number(hfnode.left, num)
            num = add_number(hfnode.right, num)
            hfnode.number = num
            num += 1
    return num


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    dict_codes = get_codes(tree)
    freq_sum = 0
    codes_sum = 0
    for symbol in freq_dict:
        freq_sum += freq_dict[symbol]
        codes_sum += freq_dict[symbol] * len(dict_codes[symbol])
    assert freq_sum != 0
    return codes_sum / freq_sum


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> d = {1:"0000", 3:"0001", 10:"001", 12:"10", 13:"11", 15:"01"}
    >>> text = bytes([1, 3, 3, 1, 10, 12, 15, 13, 15, 10, 10, 3, 1])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['00000001', '00010000', '00110011', '10100100', '10001000', '00000000']
    """
    # get bits
    bits = ''
    for symbol in text:
        bits += codes[symbol]
    # convert to bytes
    bytes_ = bytes()
    eight_bits = ''
    i = 0
    while i < len(bits):
        while len(eight_bits) < 8:
            if i < len(bits):
                eight_bits += bits[i]
                i += 1
            else:
                eight_bits += '0'
        bytes_ += bytes([bits_to_byte(eight_bits)])
        eight_bits = ''
    return bytes_


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> left = HuffmanNode(None, HuffmanNode(70), HuffmanNode(93))
    >>> right1 = HuffmanNode(None, HuffmanNode(91), HuffmanNode(69))
    >>> right2 = HuffmanNode(None, HuffmanNode(76), HuffmanNode(73))
    >>> right = HuffmanNode(None, right1, right2)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 70, 0, 93, 0, 91, 0, 69, 0, 76, 0, 73, 1, 1, 1, 2, 1, 0, 1, 3]
    """
    list_bytes = []
    list_bytes = record_hftree(tree, list_bytes)
    return bytes(list_bytes)


# helper: record tree recursively
def record_hftree(hfnode, list_bytes_):
    """ helper function for tree_to_bytes to record tree in postorder

    @param HuffmanNode hfnode:
    @param list list_bytes_:
    @rtype: list

    >>> left = HuffmanNode(None, HuffmanNode(70), HuffmanNode(93))
    >>> right1 = HuffmanNode(None, HuffmanNode(91), HuffmanNode(69))
    >>> right2 = HuffmanNode(None, HuffmanNode(76), HuffmanNode(73))
    >>> right = HuffmanNode(None, right1, right2)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list_ = []
    >>> list(record_hftree(tree, list_))
    [0, 70, 0, 93, 0, 91, 0, 69, 0, 76, 0, 73, 1, 1, 1, 2, 1, 0, 1, 3]
    """
    # check None
    if hfnode.is_leaf():
        pass
    # base case: hfnode with two leaf
    elif hfnode.left.is_leaf() and hfnode.right.is_leaf():
        list_bytes_ += [0, hfnode.left.symbol, 0, hfnode.right.symbol]
    # recursively record in post order
    else:
        record_hftree(hfnode.left, list_bytes_)
        record_hftree(hfnode.right, list_bytes_)
        if hfnode.left.is_leaf():
            list_bytes_ += [0, hfnode.left.symbol, 1, hfnode.right.number]
        elif hfnode.right.is_leaf():
            list_bytes_ += [1, hfnode.left.number, 0, hfnode.right.symbol]
        else:
            list_bytes_ += [1, hfnode.left.number, 1, hfnode.right.number]
    return list_bytes_


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """

    def generate_from_readnote(node):
        """Return a HuffmanNode that is generated from ReadNote

        @param ReadNode node: a ReadNode
        @rtype: HuffmanNode
        >>>a = ReadNode(1, 5, 0, 7)
        >>>b = generate_from_readnote(a)
        >>>b
        HuffmanNode(None, HuffmanNode(None, None, None),
         HuffmanNode(7, None, None))
        """
        final = HuffmanNode()
        if node.l_type == 0:
            final.left = HuffmanNode(node.l_data)
        elif node.l_type == 1:
            final.left = HuffmanNode()
            final.left.number = node.l_data
        if node.r_type == 0:
            final.right = HuffmanNode(node.r_data)
        elif node.r_type == 1:
            final.right = HuffmanNode()
            final.right.number = node.r_data
        return final

    answer_g = generate_from_readnote(node_lst[root_index])
    node_lst.pop(root_index)

    def rebuild(lst_in, answer):
        """Return answer which is modified according to lst_in

        @param list[ReadNode] lst_in: a list of ReadNode objects
        @param HuffmanNode answer: a tree with only a root in it
        @rtype: HuffmanNode

        >>>list_haha = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12),
        ReadNode(1, 1, 1, 0)]
        >>>root_haha = HuffmanNode(None, HuffmanNode(None, None, None),
         HuffmanNode(None, None, None))
        >>>root_haha.left.number = 1
        >>>root_haha.right.number = 0
        >>>haha = rebuild(list_haha, root_haha)
        >>>haha
        HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
        """
        if answer.left.number is None and answer.right.number is None:
            pass
        else:
            if answer.left.number is not None:
                answer.left = generate_from_readnote(lst_in[answer.left.number])
                answer.left = rebuild(lst_in, answer.left)
            if answer.right.number is not None:
                answer.right = \
                    generate_from_readnote(lst_in[answer.right.number])
                answer.right = rebuild(lst_in, answer.right)
        return answer
    return rebuild(node_lst, answer_g)


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    >>> lst = [ReadNode(0, 70, 0, 93), ReadNode(0, 91, 0, 69), \
ReadNode(0, 76, 0, 73), ReadNode(1, 1, 1, 2), ReadNode(1, 0, 1, 3)]
    >>> generate_tree_postorder(lst, 4)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(70, None, None), \
HuffmanNode(93, None, None)), HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(91, None, None), HuffmanNode(69, None, None)), \
HuffmanNode(None, HuffmanNode(76, None, None), HuffmanNode(73, None, None))))
    """
    list_node = []
    return generate_tree(node_lst, root_index, list_node)


# helper for generate_treeposter
def generate_tree(node_lst_, root_index_, list_node):
    """ helper for generate_tree_postorder to reconstruct tree

    @param list node_lst_:
    @param int root_index_:
    @param list list_node:
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 70, 0, 93), ReadNode(0, 91, 0, 69), \
ReadNode(0, 76, 0, 73), ReadNode(1, 1, 1, 2), ReadNode(1, 0, 1, 3)]
    >>> list_node = []
    >>> generate_tree(lst, 4, list_node)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(70, None, None), \
HuffmanNode(93, None, None)), HuffmanNode(None, HuffmanNode(None, \
HuffmanNode(91, None, None), HuffmanNode(69, None, None)), \
HuffmanNode(None, HuffmanNode(76, None, None), HuffmanNode(73, None, None))))
    """
    if len(node_lst_) > 0:
        current_node = node_lst_[0]
        if current_node.l_type == 0 and current_node.r_type == 0:
            new_hfnode = HuffmanNode(None, HuffmanNode(current_node.l_data),
                                     HuffmanNode(current_node.r_data))
            list_node.append(new_hfnode)
        elif current_node.l_type == 1 and current_node.r_type == 0:
            new_hfnode = HuffmanNode(None, list_node[-1],
                                     HuffmanNode(current_node.r_data))
            list_node.pop(-1)
            list_node.append(new_hfnode)
        elif current_node.l_type == 0 and current_node.r_type == 1:
            new_hfnode = HuffmanNode(None, HuffmanNode(current_node.l_data),
                                     list_node[-1])
            list_node.pop(-1)
            list_node.append(new_hfnode)
        elif current_node.l_type == 1 and current_node.r_type == 1:
            new_hfnode = HuffmanNode(None, list_node[-2], list_node[-1])
            list_node.pop(-2)
            list_node.pop(-1)
            list_node.append(new_hfnode)
        node_lst_.pop(0)
        generate_tree(node_lst_, root_index_, list_node)
        return list_node[0]


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> left = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> d = {2: '00', 3: '01', 5: '1'}
    >>> text = bytes([2, 2, 2, 3, 5, 5])
    >>> result = generate_compressed(text, d)
    >>> generate_uncompressed(tree, result, 6) == text
    True
    """
    # convert bytes text to bits
    bits = ''
    for byte in text:
        bits += byte_to_bits(byte)

    # get code dict
    code_dict = get_codes_reverse(tree)

    # generate new byte
    list_bytes = []
    count = 0
    i = 0
    bit = ''
    while i < len(bits) and count < size:
        bit += bits[i]
        if bit in code_dict:
            list_bytes.append(code_dict[bit])
            count += 1
            bit = ''
        i += 1
    return bytes(list_bytes)


# helper
def get_codes_reverse(hfnode):
    """ helper for get a dict as {code:symbol} from function get_codes

    @param HuffmanNode hfnode:
    @rtype: dict

    >>> left = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> d = get_codes_reverse(tree)
    >>> d['00']
    2
    >>> d['01']
    3
    >>> d['1']
    5
    """
    dict_to_change = get_codes(hfnode)
    new_dict = {}
    while len(dict_to_change) > 0:
        tuple_ = dict_to_change.popitem()
        new_dict[tuple_[1]] = tuple_[0]
    return new_dict


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        # tree = generate_tree_postorder(node_lst, -1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # get list of (freq, symbol)
    dict_ = freq_dict.copy()
    freq_dict_ = {}
    while len(dict_) > 0:
        tuple_ = dict_.popitem()
        freq_dict_[tuple_[1]] = tuple_[0]
    list_freq = list(freq_dict_.items())
    list_freq.sort()

    # change node based on iterative level order
    nodes = [tree]
    while nodes != []:
        current_node = nodes.pop(0)
        if current_node.is_leaf():
            current_node.symbol = list_freq[-1][-1]
            list_freq.pop(-1)
        if current_node.left is not None:
            nodes.append(current_node.left)
        if current_node.right is not None:
            nodes.append(current_node.right)

if __name__ == "__main__":

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
