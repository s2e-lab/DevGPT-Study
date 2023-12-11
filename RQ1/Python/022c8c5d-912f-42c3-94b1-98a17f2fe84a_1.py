import pandas as pd
from django.contrib import admin, messages
from django.http import HttpResponseRedirect
from django.urls import path
from django.shortcuts import render

from .models import YourModel

class YourModelAdmin(admin.ModelAdmin):
    change_list_template = "admin/your_model_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('import-csv/', self.import_csv),
        ]
        return admin.site._registry[self.model].urls

    def import_csv(self, request):
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            data_set = pd.read_csv(csv_file)
            for index, row in data_set.iterrows():
                YourModel.objects.create(name=row["Name"], title=row["Title"])
            self.message_user(request, "Your csv file has been imported")
            return HttpResponseRedirect("..")
        form = CsvImportForm()
        payload = {"form": form}
        return render(
            request, "admin/csv_form.html", payload
        )

admin.site.register(YourModel, YourModelAdmin)
