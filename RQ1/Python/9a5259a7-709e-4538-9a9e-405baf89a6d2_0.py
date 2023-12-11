from django.http import HttpResponse

def my_view(request):
    param = request.GET.get('param', 'default_value')
    return HttpResponse(f'The parameter is: {param}')
