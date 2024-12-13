from django.shortcuts import render
import tensorflow as tf
import pickle

def index(request):
    return render(request, 'index.html')

def cnn(request):
    return render(request, 'cnn.html')

def hybrid(request):
    return render(request, 'hybrid.html')

def transformer(request):
    return render(request, 'transformer.html')

def lstm(request):
    return render(request, 'lstm.html')


