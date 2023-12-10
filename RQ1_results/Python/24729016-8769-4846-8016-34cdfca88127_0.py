is_communicating = False  # グローバルなセマフォ

@api_view(['POST'])
def save_machine_data(request):
    global is_communicating
    if is_communicating:
        return Response({'message': 'Communication in progress'}, status=status.HTTP_409_CONFLICT)
    
    is_communicating = True
    # 通信処理
    is_communicating = False
