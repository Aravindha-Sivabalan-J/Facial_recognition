from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm
from deepface import DeepFace as df
from .forms import ProfileForm, CompareFacesForm, UploadForm, MultiUploadForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.files.storage import FileSystemStorage
from .models import Profile, Comparison_Images
from io import BytesIO
from django.db.models import Q
from django.conf import settings
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import os
import cv2


model_load_path = os.path.join('faceapp', 'trained_model_and_scaler', 'best_logistic_regression_model.pkl')
scaler_load_path = os.path.join('faceapp', 'trained_model_and_scaler', 'scaler.pkl')

loaded_model = None
loaded_scaler = None

try:
    loaded_model = joblib.load(model_load_path)
    print(loaded_model)
    print("Trained model loaded successfully!")
    loaded_scaler = joblib.load(scaler_load_path)
    # print(loaded_scaler)
    print("scaler loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find model or scaler files at {model_load_path} or {scaler_load_path}")
    print("Please ensure the 'trained_model_and_scaler' folder and its contents are in the correct location.")


# Create your views here.
def login_view(request):
    if request.user.is_authenticated:
        return redirect ('home_view')
    if request.method == 'POST':        
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username and password:
            if User.objects.filter(username = username).exists():
                user = authenticate(request, username=username, password=password)
                if user is not None:
                    login(request, user)
                    return redirect('home_view')
    return render(request, 'login_reg.html')

def home_view(request):
    return render(request, 'home.html')

def register_view(request):
    form=UserCreationForm()
    if request.method == "POST":
        form=UserCreationForm(request.POST)
        if form.is_valid():
            user=form.save(commit=False)
            user.save()
            login(request, user)
            return redirect ('home_view')
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login_view')

def create_profile(request):
    proform = ProfileForm()
    if request.method == 'POST':
        proform = ProfileForm(request.POST, request.FILES)
        if proform.is_valid():
            profile = proform.save(commit=False)
            if 'photo' in request.FILES:
                image_file = request.FILES['photo']
                print(type(image_file))
                image_data = image_file.read()
                print(type(image_data))
                image_stream = BytesIO(image_data)
                print(type(image_stream))
                img = Image.open(image_stream)
                print(type(img))
                img_array = np.array(img)
                print(type(img_array))
                embedding_1 = df.represent(img_path=img_array, model_name="Facenet512")
                print(type(embedding_1))
                embedding_2 = df.represent(img_path=img_array, model_name="Facenet")
                embedding_3 = df.represent(img_path=img_array, model_name="Dlib")
                embedding_4 = df.represent(img_path=img_array, model_name="VGG-Face")
                embedding_5 = df.represent(img_path=img_array, model_name="ArcFace")
                embedding_1_array = np.array(embedding_1[0]['embedding'])
                print(type(embedding_1_array))
                embedding_2_array = np.array(embedding_2[0]['embedding'])
                embedding_3_array = np.array(embedding_3[0]['embedding'])
                embedding_4_array = np.array(embedding_4[0]['embedding'])
                embedding_5_array = np.array(embedding_5[0]['embedding'])
                profile.Facenet512_embedding = embedding_1_array.tobytes()
                profile.Facenet_embedding = embedding_2_array.tobytes()
                profile.Dlib_embedding = embedding_3_array.tobytes()
                profile.VGGFace_embedding = embedding_4_array.tobytes()
                profile.ArcFace_embedding = embedding_5_array.tobytes()
            profile.save()
            return redirect ('home_view')
    context = {'proform': proform}
    return render(request, 'profileform.html', context)


def searchdb_view(request):
    context = {}
    results = []
    if request.method == 'POST':
        if 'image' in request.FILES:
            uploaded_image = request.FILES['image']
            fs = FileSystemStorage()
            temp_name = fs.save(f'temp_name_{uploaded_image.name}', uploaded_image)
            uploaded_image_path = os.path.join(settings.MEDIA_ROOT, temp_name)
            if uploaded_image_path:
                database_path = os.path.join(settings.MEDIA_ROOT, 'photos')
                try:
                    similar_images = df.find(img_path = uploaded_image_path, db_path=database_path, model_name="VGG-Face", enforce_detection=False)
                    print(f"this is the result of similar images ${similar_images}")
                    is_empty =  True
                    if isinstance(similar_images, list):
                        print("it is a list")
                        if not similar_images:
                            is_empty = True
                            print("it is an emoty list")
                            
                        else:
                            if isinstance(similar_images[0], pd.DataFrame):
                                print("it is a dataframe")
                                similar_images = similar_images[0]
                                is_empty = similar_images.empty
                                print("it is an empty dataframe")
                                
                            else:
                                context['error'] = "DeepFace returned an unexpected list format"
                                is_empty = True
                    elif isinstance(similar_images, pd.DataFrame):
                        print("it is indeed a dataframe")
                        is_empty = similar_images.empty
                        
                    else:
                        context['error'] = "DeepFace returned an unexpected type."
                        is_empty = True
                    
                    if not is_empty:
                        print("is empty is false")
                        num_rows = len(similar_images)
                        print(f"this is the num of rows in result: {num_rows}")
                        top_n = min(5, num_rows)
                        print(top_n)
                        top_matches = similar_images.head(top_n).to_dict('records')
                        for match in top_matches:
                            try:
                                print(f"these are the top matches: {top_matches}")
                                filename = os.path.basename(match['identity'])
                                match['database_img_url'] = os.path.join(settings.MEDIA_URL, 'photos', filename)
                                results.append(match)
                                print(len(results))
                            except Exception as e:
                                print(f"Error processing individual match: {e} for match: {match}")
                                context['error'] = (f"Problem processing some matches: {e}")
                                continue
                    else:
                        context['error'] = "No similar images found in the DataBase or DeepFace couldn't process the images"
                except Exception as e:
                    context['error'] = f"Error running the deepface function: {e}"
                finally:
                    fs.delete(temp_name)
            else:
                context['error'] = "image path does not exist for uploaded image"
        else:
            context['error'] = "Please upload an image."
        print("Search Results (for terminal):", results)
    context['results'] = results
    return render(request, 'searchdbform.html', context)
        


def compare_faces(request):
    all_comparison_results = []
    if request.method == 'GET':
         compareform = CompareFacesForm()
         context = {'compareform': compareform, 'results': all_comparison_results}
         return render(request, 'compareimg.html', context)

    if request.method == 'POST':
        print("POST request received")

        selected_image_paths = request.POST.getlist('selected_image_paths[]')

        if len(selected_image_paths) % 2 != 0 or len(selected_image_paths) < 2:
            print("Selected image path count is not even or less than 2")
            return JsonResponse({'error': 'Please select an even number of images (at least two) to form pairs.'}, status=400)

        for i in range(0, len(selected_image_paths), 2):
            print(f"Processing pair starting at index {i}")

            image1_relative_path = selected_image_paths[i]
            image2_relative_path = selected_image_paths[i+1]

            image1_path = os.path.join(settings.MEDIA_ROOT, image1_relative_path)
            image2_path = os.path.join(settings.MEDIA_ROOT, image2_relative_path)

            if not os.path.exists(image1_path) or not os.path.exists(image2_path):
                 print(f"File not found on server for pair {i//2 + 1}: {image1_relative_path} or {image2_relative_path}")
                 all_comparison_results.append({
                     'pair_index': i//2 + 1,
                     'image1_name': image1_relative_path,
                     'image2_name': image2_relative_path,
                     'error': 'Image file(s) not found on the server.'
                 })
                 continue

            try:
                distance_scores = []
                dummy1 = df.verify(image1_path, image2_path, model_name='Facenet512', enforce_detection=False)
                distance_scores.append(dummy1['distance'])
                dummy2 = df.verify(image1_path, image2_path, model_name='Facenet', enforce_detection=False)
                distance_scores.append(dummy2['distance'])
                dummy3 = df.verify(image1_path, image2_path, model_name='Dlib', enforce_detection=False)
                distance_scores.append(dummy3['distance'])
                dummy4 = df.verify(image1_path, image2_path, model_name='VGG-Face', enforce_detection=False)
                distance_scores.append(dummy4['distance'])
                dummy5 = df.verify(image1_path, image2_path, model_name='ArcFace', enforce_detection=False)
                distance_scores.append(dummy5['distance'])

                original_distance_array = np.array(distance_scores)

                if original_distance_array.shape[0] == 5:

                    min_distance = np.min(original_distance_array)
                    max_distance = np.max(original_distance_array)
                    mean_distance = np.mean(original_distance_array)
                    std_distance = np.std(original_distance_array)

                    input_features_for_prediction = np.hstack((original_distance_array, min_distance, max_distance, mean_distance, std_distance))
                    input_features_for_prediction = input_features_for_prediction.reshape(1,-1)

                    if loaded_scaler is not None:
                        scaled_input_for_prediction = loaded_scaler.transform(input_features_for_prediction)
                    else:
                        return JsonResponse({'error': 'Model scaler not available.'}, status=500)

                    if loaded_model is not None:
                        probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
                        probability_of_match_number = probability_of_match_array[0,1]
                    else:
                        return JsonResponse({'error': 'Prediction model not available.'}, status=500)

                    chosen_threshold = 0.499
                    final_decision = "Not a match"
                    if probability_of_match_number >= chosen_threshold:
                        final_decision = "Match"
                    print(final_decision)

                    all_comparison_results.append({
                        "pair_index": i//2+1,
                        "image1_name": image1_relative_path,
                        "image2_name": image2_relative_path,
                        "final_decision": final_decision,
                        'confidence_score': f"{probability_of_match_number:.4f}",
                        'original_distances': original_distance_array.tolist()
                    })
                    print(f"Finished pair {i//2 + 1}")

                else:
                    print(f"Could not get 5 distance scores for pair {i//2 + 1} ({image1_relative_path}, {image2_relative_path}).")
                    all_comparison_results.append({
                       'pair_index': i//2 + 1,
                       'image1_name': image1_relative_path,
                       'image2_name': image2_relative_path,
                       'error': 'Could not get 5 distance scores.'
                    })
            except Exception as e:
                print(f"An unexpected error occurred processing pair {i//2+1} ({image1_relative_path}, {image2_relative_path}): {e}")
                all_comparison_results.append({
                    'pair_index': i//2 + 1,
                    'image1_name': image1_relative_path,
                    'image2_name': image2_relative_path,
                    'error': f"An unexpected error occurred: {e}"
                })
        return JsonResponse({'results': all_comparison_results})
    else:
        compareform = CompareFacesForm()
        context = {'compareform': compareform, 'results': all_comparison_results}
        return render(request, 'compareimg.html', context)


# def list_images(request):
#     images_from_db = Comparison_Images.objects.all()
#     image_data=[]
#     for image_obj in images_from_db:
#         image_data.append({
#             'id': image_obj.id,
#             'name': image_obj.images.name,
#             'image': image_obj.images.url,
#         })
#     return JsonResponse({'images': image_data})


def list_images(request):
    images_from_db = Comparison_Images.objects.all().order_by('id')

    page_size = int(request.GET.get('limit', 100))
    page_number = request.GET.get('page', 1)

    paginator = Paginator(images_from_db, page_size)

    try:
        current_page_images = paginator.page(page_number)
    except PageNotAnInteger:
        current_page_images = paginator.page(1)
    except EmptyPage:
        current_page_images = paginator.page(paginator.num_pages)

    image_data = []
    for image_obj in current_page_images:
        image_data.append({
            'id': image_obj.id,
            'name': image_obj.images.name, 
            'image': image_obj.images.url,
        })

    return JsonResponse({
        'images': image_data,
        'total_images': paginator.count,       
        'total_pages': paginator.num_pages,     
        'current_page': current_page_images.number,
        'page_size': page_size
    })


def upload_view(request):
    upload = UploadForm()
    if request.method == 'POST':
        upload = UploadForm(request.POST, request.FILES)
        if upload.is_valid():
            comparison_image = upload.save(commit=False)
            comparison_image.save()
            context = {'upload':upload}
            return render(request, 'uploadform.html', context)
    context = {'upload':upload}
    return render(request, 'uploadform.html', context)

def multiupload_view(request):
    multiupload = MultiUploadForm()
    if request.method == 'POST':
        multiupload = MultiUploadForm(request.POST, request.FILES)
        if multiupload.is_valid():
            uploaded_images = request.FILES.getlist('images_to_be_uploaded')
            for image in uploaded_images:
                comparison_image = Comparison_Images()
                comparison_image.images = image
                comparison_image.save()
    multiupload = MultiUploadForm()
    context = {'multiupload':multiupload}
    return render(request, 'uploadform.html', context)
    

def video_scanner(request):
    matches = []
    snapshots = []
    
    unique_persons = set() 

    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            fs = FileSystemStorage()
            filename = fs.save(video.name, video)
            video_path = fs.path(filename)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = int(fps * 1)
            BLACK_FRAME_THRESHOLD = 15

            print(f"Video FPS: {fps}")
            print(f"Total Frames: {total_frames}")
            print(f"Snapshot Interval (frames): {interval}")
            print(f"Expected number of snapshots: {total_frames / interval}")

            frame_num = 0 
            snapshot_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                current_frame_index = frame_num 
                frame_num += 1 
                print(f"Processing Frame {current_frame_index}: ret={ret}") 
                if not ret:
                    print(f"Breaking loop at frame {current_frame_index} because ret is False.")
                    break                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray_frame)
                if mean_intensity < BLACK_FRAME_THRESHOLD:
                    print(f"Skipping processing for frame {current_frame_index} due to low intensity ({mean_intensity:.2f}) - too dark/black.") 
                    continue
                if current_frame_index % interval == 0:
                    snapshot_path = os.path.join(settings.MEDIA_ROOT, f'snapshot_{snapshot_index}.jpg')
                    cv2.imwrite(snapshot_path, frame)
                    try:
                        results = df.find(
                            img_path=snapshot_path, 
                            db_path=os.path.join(settings.MEDIA_ROOT, 'photos'), 
                            # model_name="ArcFace",          
                            # distance_metric="euclidean_l2", 
                            # detector_backend="retinaface",
                            enforce_detection=False
                        )
                        print("Results for snapshot:", snapshot_path)
                        print("Returned results:", results) 

                        if results and not results[0].empty:
                            confirm_matches_for_current_snapshot = [] 
                            similar_matches_for_current_snapshot = []
                            STRICT_THRESHOLD = 0.45
                            LOOSE_THRESHOLD = 0.61
                            
                            has_new_person_for_display = False

                            for i, row in results[0].iterrows():
                                person_identity_path = row['identity']
                                current_distance = row['distance']
                                print(f"  Match candidate: {person_identity_path}, Distance: {current_distance:.4f}") 
                                if current_distance <= STRICT_THRESHOLD:
                                    rel_path = os.path.relpath(person_identity_path, settings.BASE_DIR).replace("\\", "/")
                                    confirm_matches_for_current_snapshot.append({
                                        'image': rel_path,
                                        'distance': current_distance,
                                        'status': 'confirmed match'
                                    })
                                    if person_identity_path not in unique_persons:
                                        has_new_person_for_display = True
                                        
                                elif current_distance <= LOOSE_THRESHOLD:
                                    rel_path = os.path.relpath(person_identity_path, settings.BASE_DIR).replace("\\","/")
                                    similar_matches_for_current_snapshot.append({
                                        'image': rel_path,
                                        'distance': current_distance,
                                        'status': 'similar profile'
                                    })
                                    if person_identity_path not in unique_persons:
                                        has_new_person_for_display = True
                                else:
                                    print(f"  Skipping match for {person_identity_path} due to high distance: {current_distance:.4f} (above {LOOSE_THRESHOLD})") 
                            if has_new_person_for_display:
                                for match_item in confirm_matches_for_current_snapshot:
                                    unique_persons.add(os.path.join(settings.BASE_DIR, match_item['image']).replace("/", os.sep))
                                for match_item in similar_matches_for_current_snapshot:
                                    unique_persons.add(os.path.join(settings.BASE_DIR, match_item['image']).replace("/", os.sep))

                                snapshot_rel_path = f'media/snapshot_{snapshot_index}.jpg'
                                snapshots.append(snapshot_rel_path)
                                if confirm_matches_for_current_snapshot:
                                    matches.append(confirm_matches_for_current_snapshot)
                                    print(f"Snapshot {snapshot_index}: {len(confirm_matches_for_current_snapshot)} confirmed match(es) added (some might be new for display).")
                                elif similar_matches_for_current_snapshot:
                                    matches.append(similar_matches_for_current_snapshot)
                                    print(f"Snapshot {snapshot_index}: No confirmed, but {len(similar_matches_for_current_snapshot)} similar match(es) added (some might be new for display).")
                                else:
                                    print(f"Snapshot {snapshot_index}: New person(s) detected, but no matches met thresholds for display.")
                            else:
                                print(f"Snapshot {snapshot_index}: No new unique persons found for display. Skipping adding to final lists.")
                        else:
                            print(f"No faces detected or no matches returned by DeepFace for snapshot: {snapshot_index}")

                    except Exception as e:
                        print(f"DeepFace error for snapshot {snapshot_index}: {e}")

                    snapshot_index += 1 

            cap.release()
            os.remove(video_path)

    context = {
        'matches': matches,
        'snapshots': snapshots,
        'combined': zip(snapshots, matches)
    }

    return render(request, 'videoscanning.html', context)