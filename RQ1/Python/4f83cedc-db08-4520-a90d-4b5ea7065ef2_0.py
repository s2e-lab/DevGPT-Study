ALLOWED_CHARS = set([*b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?#$%&-_"])
Y = [model.NewBoolVar(f"y{i}") for i in range(160)]

linear_expr = Y[i+1]*64 + Y[i+2]*32 + Y[i+3]*16 + Y[i+4]*8 + Y[i+5]*4 + Y[i+6]*2 + Y[i+7]*1

is_allowed = model.NewBoolVar("is_allowed")
is_char = [model.NewBoolVar(f"is_char_{char}") for char in ALLOWED_CHARS]

for char, var in zip(ALLOWED_CHARS, is_char):
    model.Add(linear_expr == char).OnlyEnforceIf(var)

model.AddBoolOr(is_char).OnlyEnforceIf(is_allowed)
model.Add(is_allowed)
