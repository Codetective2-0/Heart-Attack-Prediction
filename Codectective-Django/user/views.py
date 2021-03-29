from django.shortcuts import render

# Create your views here.
from user.models import Contact

def contact(request):
    if request.method=="POST":
        name=request.POST['name']
        email=request.POST['email']
        subject = request.POST['subject']
        content =request.POST['content']
        contact=Contact(name=name, email=email, subject=subject, content=content)
        contact.save()
        return render(request, 'index.html',{})
    else:
        return render(request, 'index.html',{})