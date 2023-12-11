def lptr(rank1, rank2, ap_rank1, ap_rank2):
    diff_tournament = abs(rank1 - rank2)
    diff_ap = abs(ap_rank1 - ap_rank2)

    tournament_weight = 0.7  # Weight for tournament ranking
    ap_weight = 0.3  # Weight for AP ranking

    y_tournament = (0.5 / 15) * diff_tournament + 0.5
    y_ap = (0.5 / 25) * diff_ap + 0.5

    # Combine the probabilities based on weights
    weighted_probability_1 = tournament_weight * y_tournament + ap_weight * y_ap
    weighted_probability_2 = 1 - weighted_probability_1

    return weighted_probability_1, weighted_probability_2
