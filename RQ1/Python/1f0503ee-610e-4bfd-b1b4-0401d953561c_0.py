def next_braille_representation(current_rep):
    # Convert the current representation to an integer (treated as binary)
    current_int = int(current_rep, 2)

    # Calculate the next integer representation
    next_int = ((current_int & 0b001111) << 1) | 0b000001

    # Convert the next integer back to a 6-bit binary string
    next_rep = format(next_int, '06b')

    return next_rep
