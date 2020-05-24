from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .process import *
# Create your views here.
def index(request):
    if (request.method == "POST"):
        if (request.POST.get("iddata") == "datacsv"):
            upload_data = request.FILES['datacsv']
            fs = FileSystemStorage()
            fs.delete("dataset.csv")
            fs.save("dataset.csv", upload_data)
        if (request.POST.get("iddata") == "no_to_num"):
            non_to_num()
        if (request.POST.get("iddata") == "missing_value"):
            deadling_missing_value()
        if (request.POST.get("iddata") == "extract_feature"):
            print(request.POST.get("iddata"))
        if (request.POST.get("iddata") == "training"):
            training()
        if (request.POST.get("iddata") == "data_pattern"):
            print(request.POST.get("iddata"))
        if (request.POST.get("iddata") == "data_patterns"):
            print(request.POST.get("iddata"))
            test_data = request.FILES['testcsv']
            fs = FileSystemStorage()
            fs.delete("testset.csv")
            fs.save("testset.csv", upload_data)
    return render(request, 'home/index.html')
