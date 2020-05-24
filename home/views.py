from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.
def index(request):
    if (request.method == "POST"):
        if (request.POST.get("iddata") == "datacsv"):
            upload_data = request.FILES['datacsv']
            fs = FileSystemStorage()
            fs.delete("dataset.csv")
            fs.save("dataset.csv", upload_data)
        if (request.POST.get("iddata") == "no_to_num"):
            no_to_num()
        if (request.POST.get("iddata") == "missing_value"):
            missing_value()
        if (request.POST.get("iddata") == "extract_feature"):
            extract_feature()
        if (request.POST.get("iddata") == "training"):
            training()
        if (request.POST.get("iddata") == "data_pattern"):
            data_pattern()
        if (request.POST.get("iddata") == "data_patterns"):
            data_patterns()
    return render(request, 'home/index.html', {'iddata': request.POST.get("iddata")})

def no_to_num():
    print(1)
def missing_value():
    print(2)
def extract_feature():
    print(3)
def training():
    print(4)
def data_pattern():
    print(5)
def data_patterns():
    test_data = request.FILES['testcsv']
    fs = FileSystemStorage()
    fs.delete("testset.csv")
    fs.save("testset.csv", test_data)

def visualization(request):
    return render(request, 'home/visualization.html')