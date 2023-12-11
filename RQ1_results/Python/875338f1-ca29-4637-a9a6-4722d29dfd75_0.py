def sigmodal(rank1, rank2, k=1):
    """
    Sigmoidal Ranking Probabilities (Adjusted)
    ------------------------------------------

    This function calculates the winning probabilities between two teams based on their rankings using an adjusted sigmoidal function.
    The adjusted sigmoidal function ensures that the team with the higher ranking always has a higher probability of winning,
    regardless of the rank difference between the teams.

    Parameters:
    -----------
    rank1 : int
        The ranking of the first team.
    rank2 : int
        The ranking of the second team.
    k : float, optional
        The scaling factor controlling the steepness of the sigmoidal curve. Default is 1.

    Returns:
    --------
    tuple
        A tuple containing the winning probability of the first team and the winning probability of the second team.
        The probabilities are not normalized and sum up to 1.

    Example:
    --------
    Here is an example usage of the function:
    ```python
    >>> sigmodal(8, 9, k=0.5)
    (0.5634215777379688, 0.4365784222620312)
    ```

    Note:
    -----
    The adjusted sigmoidal ranking probabilities function guarantees that the team with the higher ranking will always have a higher
    probability of winning. This ensures a consistent ranking behavior, regardless of the magnitude of the rank difference between the teams.
    The steepness of the curve is controlled by the "k" parameter, allowing for customization based on specific requirements.
    """
    # Function implementation here
    pass
