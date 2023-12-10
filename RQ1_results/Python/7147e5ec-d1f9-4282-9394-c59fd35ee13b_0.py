mul(
    mul(
        -1, 
        cs_rank(ts_rank('self.close', 10))
    ), 
    cs_rank(divide('self.close', 'self.open'))
)
