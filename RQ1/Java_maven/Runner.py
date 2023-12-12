# %%
import subprocess


# %%
project_names = """
0514eecc-9214-49db-9598-093dbbb5a588_0
08649025-20e3-42e6-abba-32c29722dc0c_0
0a2bd87e-2797-4067-8290-45c5381f39bf_0
0a69776f-09a0-492f-94e9-97ca649fa1c2_0
12b4c378-3a0a-478b-aa99-280771801977_0
17e938b3-a7d4-42f1-ba1f-b3186df65836_3
2141682a-9089-4e5e-bfc8-790dd16f68f0_0
2141682a-9089-4e5e-bfc8-790dd16f68f0_1
2141682a-9089-4e5e-bfc8-790dd16f68f0_2
21e49b75-516f-46bd-8cf7-2eeffac7bb97_0
25819bff-f407-46dc-9b52-62d4976293c0_0
2770bf7b-9927-4fd5-a675-e34dce7cefc7_0
2770bf7b-9927-4fd5-a675-e34dce7cefc7_1
2770bf7b-9927-4fd5-a675-e34dce7cefc7_3
2770bf7b-9927-4fd5-a675-e34dce7cefc7_4
2a8f82bb-ce20-496d-8ddc-b5df99705ff6_1
2b55d53f-c2df-4f92-abf2-437cd3c7a0e6_0
2f8bded2-4b85-4784-9449-bdda58b030d1_0
3069aebe-87ad-4354-a839-5c29f864f9e2_0
336b057b-3049-4246-a9d7-77231cc89785_2
336b057b-3049-4246-a9d7-77231cc89785_4
3f0b59f3-d7ce-4d88-a6a6-da3056da4128_0
52305262-66c1-4b11-92e0-b81694dae809_0
60cb28bf-183b-4448-aef5-aad5b56e40fc_0
60cb28bf-183b-4448-aef5-aad5b56e40fc_1
60cb28bf-183b-4448-aef5-aad5b56e40fc_2
60cb28bf-183b-4448-aef5-aad5b56e40fc_3
633e2b6f-6e7b-43e7-8a8a-7fbe47c0aa4b_0
7769ab8b-6f90-45d4-affa-4902a5e4d4ee_0
85a33a1a-8b2b-4942-b82a-317946dfd425_0
85a33a1a-8b2b-4942-b82a-317946dfd425_2
85a33a1a-8b2b-4942-b82a-317946dfd425_3
89737d1b-08e5-452c-a2e8-d59c58b59a3f_0
db090593-45ec-4f1e-99d9-d0bf66b98519_0
db090593-45ec-4f1e-99d9-d0bf66b98519_1
db090593-45ec-4f1e-99d9-d0bf66b98519_2
db090593-45ec-4f1e-99d9-d0bf66b98519_3
dd82b2ad-dd07-40dc-bd65-52bd3b44ef94_0
dd82b2ad-dd07-40dc-bd65-52bd3b44ef94_3
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_0
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_1
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_2
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_3
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_4
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_5
e91b3827-8518-4656-b4c4-79d1bbdb3f4d_6
e9b335ea-b23e-49f9-9a31-ea5009817ed3_1
fdf1e5e2-4cfa-485f-a00b-a179e1a0cacc_0
fdf1e5e2-4cfa-485f-a00b-a179e1a0cacc_1
fdf1e5e2-4cfa-485f-a00b-a179e1a0cacc_2
fe6258ed-93d6-446b-a39b-6b9c9d1ac0b1_0
fe6258ed-93d6-446b-a39b-6b9c9d1ac0b1_1
fe6258ed-93d6-446b-a39b-6b9c9d1ac0b1_2
fe6258ed-93d6-446b-a39b-6b9c9d1ac0b1_3
"""



# %%
for project_name in project_names.split():
    with open('./script_bk.sh') as f:
        content = f.read()

    content = content.replace('PROJECT_NAME', project_name)
    content = content.replace('_DB_ID', '')

    with open('./script.sh', 'w') as f:
        f.write(content)

    subprocess.run(['bash', './script.sh'])


# %%



