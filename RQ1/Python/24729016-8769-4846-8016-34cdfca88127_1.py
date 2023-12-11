from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def save_machine_data(request):
    if request.method == 'POST':
        plc_data = request.data  # PLCから送られてきたデータを受け取る
        machine_data = MachineData.create_from_plc_data(plc_data)
        if machine_data:
            return Response({'message': 'Data successfully saved'}, status=status.HTTP_201_CREATED)
        else:
            return Response({'message': 'Failed to save data'}, status=status.HTTP_400_BAD_REQUEST)
