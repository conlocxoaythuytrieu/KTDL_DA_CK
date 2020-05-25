from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .process import *
from .training import *
# Create your views here.
def index(request):
    if (request.method == "POST"):
        if (request.POST.get("iddata") == "datacsv"):
            upload_data = request.FILES['datacsv']
            fs = FileSystemStorage()
            fs.delete("dataset.csv")
            fs.save("dataset.csv", upload_data)
        if (request.POST.get("iddata") == "no_to_num"):
            dataset_numerical_train()
        if (request.POST.get("iddata") == "missing_value"):
            deadling_missing_value_train()
        if (request.POST.get("iddata") == "extract_feature"):
            feture_extraction()
        if (request.POST.get("iddata") == "training"):
            training()
        if (request.POST.get("iddata") == "data_pattern"):
            result = train_data_pattern(request.POST.get("data_pattern"))
            print(result)
        if (request.POST.get("iddata") == "data_patterns"):
            data_patterns(request)
            result = train_data_patterns()
            print(result)
    return render(request, 'home/index.html', {'iddata': request.POST.get("iddata")})



def data_patterns(request):
    test_data = request.FILES['testcsv']
    fs = FileSystemStorage()
    fs.delete("testset.csv")
    fs.save("testset.csv", test_data)

def visualization(request):
    return render(request, 'home/visualization.html')