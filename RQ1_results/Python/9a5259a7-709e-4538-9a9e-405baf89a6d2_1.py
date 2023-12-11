from django.http import HttpResponse
from django.views import View

class MyView(View):
    def get(self, request):
        param = request.GET.get('param', 'default_value')
        return HttpResponse(f'The parameter is: {param}')
