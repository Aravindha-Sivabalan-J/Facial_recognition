from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm
from deepface import DeepFace as df
from .forms import ProfileForm, CompareFacesForm
from django.core.files.storage import FileSystemStorage
from .models import Profile
from io import BytesIO
from django.db.models import Q
from django.conf import settings
from PIL import Image
import numpy as np
import joblib
import os
import cv2
from .utils import compare_faces, cosine_similarity, compare_face

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
    similarities = []
    if request.method == 'POST':
        if 'image' in request.FILES:
            uploaded_image= request.FILES['image']
            image_data = uploaded_image.read()
            image_stream = BytesIO(image_data)
            img = Image.open(image_stream)
            img_array = np.array(img)
            uploaded_1 = df.represent(img_path=img_array, model_name="Facenet512")
            uploaded_2 = df.represent(img_path=img_array, model_name="Facenet")
            uploaded_3 = df.represent(img_path=img_array, model_name="Dlib")
            uploaded_4 = df.represent(img_path=img_array, model_name="VGG-Face")
            uploaded_5 = df.represent(img_path=img_array, model_name="ArcFace")
            uploaded_embedding_1 = np.array(uploaded_1[0]['embedding']) if uploaded_1 else None
            uploaded_embedding_2 = np.array(uploaded_2[0]['embedding']) if uploaded_2 else None
            uploaded_embedding_3 = np.array(uploaded_3[0]['embedding']) if uploaded_3 else None
            uploaded_embedding_4 = np.array(uploaded_4[0]['embedding']) if uploaded_4 else None
            uploaded_embedding_5 = np.array(uploaded_5[0]['embedding']) if uploaded_5 else None
            
            pro_with_enc = Profile.objects.exclude(Q(Facenet512_embedding__isnull=True) | 
                                                Q(Facenet_embedding__isnull=True) | 
                                                Q(Dlib_embedding__isnull=True) | 
                                                Q(VGGFace_embedding__isnull=True) | 
                                                Q(ArcFace_embedding__isnull=True))
            for profile in pro_with_enc:
                stored_Facenet512 = np.frombuffer(profile.Facenet512_embedding, dtype=np.float64) if profile.Facenet512_embedding else None
                stored_Facenet = np.frombuffer(profile.Facenet_embedding, dtype=np.float64) if profile.Facenet_embedding else None
                stored_Dlib = np.frombuffer(profile.Dlib_embedding, dtype=np.float64) if profile.Dlib_embedding else None
                stored_VGGFace = np.frombuffer(profile.VGGFace_embedding, dtype=np.float64) if profile.VGGFace_embedding else None
                stored_ArcFace = np.frombuffer(profile.ArcFace_embedding, dtype=np.float64) if profile.ArcFace_embedding else None
            
            profile_similarities = {}

            if uploaded_embedding_1 is not None and stored_Facenet512 is not None:
                print(f"uploaded_embedding_1 shape: {uploaded_embedding_1.shape}, stored_Facenet512 shape: {stored_Facenet512.shape}")
                profile_similarities['Facenet512'] = float(compare_faces(uploaded_embedding_1, stored_Facenet512))
            if uploaded_embedding_2 is not None and stored_Facenet is not None:
                print(f"uploaded_embedding_2 shape: {uploaded_embedding_2.shape}, stored_Facenet shape: {stored_Facenet.shape}")
                profile_similarities['Facenet'] = float(compare_faces(uploaded_embedding_2, stored_Facenet))
            if uploaded_embedding_3 is not None and stored_Dlib is not None:
                print(f"uploaded_embedding_3 shape: {uploaded_embedding_3.shape}, stored_Dlib shape: {stored_Dlib.shape}")
                profile_similarities['Dlib'] = float(compare_faces(uploaded_embedding_3, stored_Dlib))
            if uploaded_embedding_4 is not None and stored_VGGFace is not None:
                print(f"uploaded_embedding_4 shape: {uploaded_embedding_4.shape}, stored_VGGFace shape: {stored_VGGFace.shape}")
                profile_similarities['VGG-Face'] = float(compare_faces(uploaded_embedding_4, stored_VGGFace))
            if uploaded_embedding_5 is not None and stored_ArcFace is not None:
                print(f"uploaded_embedding_5 shape: {uploaded_embedding_5.shape}, stored_ArcFace shape: {stored_ArcFace.shape}")
                profile_similarities['ArcFace'] = float(compare_faces(uploaded_embedding_5, stored_ArcFace))
            similarities.append({
                'profile.id': profile.id,
                'similarities': profile_similarities,
            })
            print(similarities)
        
    context['similarities'] = similarities    
    return render(request, 'searchdbform.html', context)


def compare_faces(request):
    compareform = CompareFacesForm(request.POST, request.FILES)
    all_comparison_results = []
    if request.method == 'POST':
        print("step 1 passed")
        compareform = CompareFacesForm(request.POST, request.FILES)
        if compareform.is_valid():
            print("form is valid")
            uploaded_files = request.FILES.getlist('images')

            if len(uploaded_files)%2 != 0 or len(uploaded_files)<2:
                print("uploaded length is not 2 files")
                context = {'compareform': compareform, 'error': 'Please upload an even number of images (at least two) to form pairs.'}
                return render(request, 'compareimg.html', context)

            # --- START OF THE LOOP THAT PROCESSES PAIRS ---
            for i in range(0, len(uploaded_files), 2):
                print(i)
                image_1 = uploaded_files[i]
                image_2 = uploaded_files[i+1]
                temp_image1_filename = f'temp_upload_{request.user.id or "anon"}_{i}_1_{image_1.name}'
                temp_image2_filename = f'temp_upload_{request.user.id or "anon"}_{i}_2_{image_2.name}'
                image_1_path = os.path.join(settings.MEDIA_ROOT, temp_image1_filename) # Corrected from literal string
                image_2_path = os.path.join(settings.MEDIA_ROOT, temp_image2_filename) # Corrected from literal string

                try:
                    with open(image_1_path, 'wb+')as destination:
                        for chunk in image_1.chunks():
                            destination.write(chunk)
                    with open(image_2_path, 'wb+')as destination:
                        for chunk in image_2.chunks(): # <-- BUG: Was writing image_1.chunks() here again
                            destination.write(chunk)

                    distance_scores = []
                    try:
                        # This check 'if image_1_path is not None' is redundant here
                        # The try/except around the deepface calls is sufficient
                        # if image_1_path is not None and image_2_path is not None:

                        dummy1 = df.verify(image_1_path, image_2_path, model_name='Facenet512', enforce_detection=False)
                        distance_scores.append(dummy1['distance'])
                        dummy2 = df.verify(image_1_path, image_2_path, model_name='Facenet', enforce_detection=False)
                        distance_scores.append(dummy2['distance']) # <-- BUG: Should append dummy2['distance']
                        dummy3 = df.verify(image_1_path, image_2_path, model_name='Dlib', enforce_detection=False)
                        distance_scores.append(dummy3['distance']) # <-- BUG: Should append dummy3['distance']
                        dummy4 = df.verify(image_1_path, image_2_path, model_name='ArcFace', enforce_detection=False) # <-- BUG: Was VGG-Face in original comment, ArcFace in code
                        distance_scores.append(dummy4['distance']) # <-- BUG: Should append dummy4['distance']
                        dummy5 = df.verify(image_1_path, image_2_path, model_name='VGG-Face', enforce_detection=False) # <-- BUG: Was ArcFace in original comment, VGG-Face in code
                        distance_scores.append(dummy5['distance']) # <-- BUG: Should append dummy5['distance']
                        # else:
                        #    context = {'compareform': compareform, 'error': 'either one of the file path missing'}
                        #    return render(request, 'compareimg.html', context) # <-- This return would stop ALL processing

                    # FIX: Use 'Exception'
                    except Exception as e:
                        print(f"DeepFace verification failed for pair {i//2 + 1}: {e}")
                        all_comparison_results.append({
                                 'pair_index': i//2 + 1, # Pair number (starting from 1)
                                 # BUG: temp_image1_file and temp_image2_file do not exist. Should be image_1.name and image_2.name
                                 'image1_name': image_1.name,
                                 'image2_name': image_2.name,
                                 'error': f'Verification failed: {e}'
                             })
                        continue # Skip the rest of this loop iteration for this pair

                    # FIX: Corrected variable name
                    original_distance_array = np.array(distance_scores)

                    # Check if we got exactly 5 distances
                    if original_distance_array.shape[0] == 5:
                        # FIX: Corrected variable names
                        min_distance = np.min(original_distance_array)
                        max_distance = np.max(original_distance_array)
                        mean_distance = np.mean(original_distance_array)
                        std_distance = np.std(original_distance_array)

                        # FIX: Corrected variable name AND np.hstack needs args in a tuple
                        input_features_for_prediction = np.hstack((original_distance_array, min_distance, max_distance, mean_distance, std_distance))
                        # FIX: Corrected variable name
                        input_features_for_prediction = input_features_for_prediction.reshape(1,-1)

                        if loaded_scaler is not None:
                            # FIX: Corrected variable name
                            scaled_input_for_prediction = loaded_scaler.transform(input_features_for_prediction)
                        else:
                            context = {'compareform': compareform, 'error': 'scaler not available.'}
                            # This return would stop ALL processing immediately
                            return render(request, 'compareimg.html', context)

                        if loaded_model is not None:
                            # FIX: Corrected variable name
                            probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
                            probability_of_match_number = probability_of_match_array[0,1]
                        else:
                            context = {'compareform': compareform, 'error': 'Model not available.'}
                            # This return would stop ALL processing immediately
                            return render(request, 'compareimg.html', context)

                        chosen_threshold = 0.499
                        final_decision = "Not a match"
                        if probability_of_match_number >= chosen_threshold:
                            final_decision = "Match"
                            print(final_decision) # For debugging

                        # Append results for this pair
                        all_comparison_results.append({
                            "pair_index": i//2+1, # Use consistent key name
                            # BUG: Use image_1.name and image_2.name for original filenames
                            "image1_name": temp_image1_filename,
                            "image2_name": temp_image2_filename, # FIX: Corrected duplicate key name
                            "final_decision": final_decision,
                            'confidence_score': f"{probability_of_match_number:.4f}",
                            'original_distances': original_distance_array.tolist()
                        })
                        print(all_comparison_results) # For debugging

                    # --- BUG: This else block is misplaced ---
                    # It's currently the 'else' for the threshold check (if probability >= threshold)
                    # It should be the 'else' for 'if original_distance_array.shape[0] == 5:'
                    else:
                        print(f"Could not get 5 distance scores for pair {i//2 + 1}.")
                        all_comparison_results.append({
                            'pair_index': i//2 + 1,
                             # BUG: Use image_1.name and image_2.name
                            'image1_name': temp_image1_file.name,
                            'image2_name': temp_image2_file.name,
                            'error': 'Could not get 5 distance scores.'
                         })

                # FIX: Use 'Exception'
                except Exception as e:
                    print(f"an error occured while processing pair {i//2+1}: {e}")
                    all_comparison_results.append({
                         'pair_index': i//2 + 1, # Pair number (starting from 1)
                          # BUG: Use image_1.name and image_2.name
                         'image1_name': temp_image1_file.name,
                         'image2_name': temp_image2_file.name,
                         'error': f"an error occured: {e}"
                     })

                finally:
                    # Clean up temporary files for the CURRENT pair
                    # BUG: 'image1_path' and 'image2_path' variables are correct in the 'os.remove' calls
                    # but they were previously constructed using literal strings (now fixed above).
                    # The os.path.exists checks here are using the correct variable names.
                    os.remove(image_1_path) if os.path.exists(image_1_path) else None # FIX: variable name mismatch in os.path.exists
                    os.remove(image_2_path) if os.path.exists(image_2_path) else None # FIX: variable name mismatch in os.path.exists

                # --- MAJOR BUG HERE ---
                # This return statement is INSIDE the for loop.
                # It executes after the first pair is processed (or errors out in a specific way).
                # This causes the view to stop and render the template showing results/errors for only the first pair.
            context = {'compareform': compareform, 'results': all_comparison_results}
            return render(request, 'compareimg.html', context)
        # If form is NOT valid
        else:
            print("form is not valid for submission")
            print(compareform.errors)
            context = {'compareform': compareform}
            return render(request, 'compareimg.html', context)

    # --- If request method is GET ---
    else:
        context = {'compareform': compareform}
        return render(request, 'compareimg.html', context)

# --- END OF FUNCTION ---


# def compare_faces(request):
#     all_comparison_results = []
#     compareform = CompareFacesForm(request.POST, request.FILES)
#     if request.method == 'POST':
#         if compareform.is_valid():
#             uploaded_files = request.FILES.getlist('images')
#             if len(uploaded_files)%2 != 0 or len(uploaded_files)<2:
#                 context = {'compareform': compareform, 'error': 'Please upload an even number of images (at least two) to form pairs.'}
#                 return render(request, 'compareimg.html', context)
#             for i in range(0, len(uploaded_files), 2):
#                 image_1 = uploaded_files[i]
#                 image_2 = uploaded_files[i+1]
#                 temp_image1_filename = f'temp_upload_{request.user.id or "anon"}_{i}_1_{image_1.name}'
#                 temp_image2_filename = f'temp_upload_{request.user.id or "anon"}_{i}_2_{image_2.name}'
#                 image_1_path = os.path.join(settings.MEDIA_ROOT, temp_image1_filename)
#                 image_2_path = os.path.join(settings.MEDIA_ROOT, temp_image2_filename)
#                 try:
#                     with open(image_1_path, 'wb+')as destination:
#                         for chunk in image_1.chunks():
#                             destination.write(chunk)
#                     with open(image_2_path, 'wb+')as destination:
#                         for chunk in image_2.chunks():
#                             destination.write(chunk)
#                     distance_scores = []
#                     try:
#                         if image_1_path is not None and image_2_path is not None:
#                             dummy1 = df.verify(image_1_path, image_2_path, model_name='Facenet512', enforce_detection=False)
#                             distance_scores.append(dummy1['distance'])
#                             dummy2 = df.verify(image_1_path, image_2_path, model_name='Facenet', enforce_detection=False)
#                             distance_scores.append(dummy1['distance'])
#                             dummy3 = df.verify(image_1_path, image_2_path, model_name='Dlib', enforce_detection=False)
#                             distance_scores.append(dummy1['distance'])
#                             dummy4 = df.verify(image_1_path, image_2_path, model_name='ArcFace', enforce_detection=False)
#                             distance_scores.append(dummy1['distance'])
#                             dummy5 = df.verify(image_1_path, image_2_path, model_name='VGG-Face', enforce_detection=False)
#                             distance_scores.append(dummy1['distance'])
#                     except Exception as e:
#                         print(f"DeepFace verification failed for pair {i//2 + 1}: {e}")
#                         all_comparison_results.append({
#                                 'pair_index': i//2 + 1, # Pair number (starting from 1)
#                                 'image1_name': temp_image1_file.name,
#                                 'image2_name': temp_image2_file.name,
#                                 'error': f'Verification failed: {e}'
#                             })
#                         continue

#                     original_distance_array = np.array(distance_scores)

#                     if original_distance_array.shape[0] == 5:
#                         min_distance = np.min(original_distance_array)
#                         max_distance = np.max(original_distance_array)
#                         mean_distance = np.mean(original_distance_array)
#                         std_distance = np.std(original_distance_array)

#                         input_features_for_prediction = np.hstack((original_distance_array, min_distance, max_distance, mean_distance, std_distance))
#                         input_features_for_prediction = input_features_for_prediction.reshape(1,-1)   

#                         if loaded_scaler is not None:
#                             scaled_input_for_prediction = loaded_scaler.transform(input_features_for_prediction)
#                         else:
#                             context = {'compareform': compareform, 'error': 'scaler not available.'}
#                             return render(request, 'compareimg.html', context)

#                         if loaded_model is not None:
#                             probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
#                             probability_of_match_number = probability_of_match_array[0,1]
#                         else:
#                             context = {'compareform': compareform, 'error': 'Model not available.'}
#                             return render(request, 'compareimg.html', context)


#                         chosen_threshold = 0.48
#                         final_decision = "Not a match"
#                         if probability_of_match_number >= chosen_threshold:
#                             final_decision = "Match"
#                             print(final_decision)
#                         all_comparison_results.append({
#                             "image_index": i//2+1,
#                             "image_1_name": temp_image1_filename,
#                             "image_2_name": temp_image2_filename,
#                             "final_decision": final_decision,
#                             'confidence_score': f"{probability_of_match_number:.4f}",
#                             'original_distances': original_distance_array.tolist()
#                         })
#                         print(all_comparison_results)
#                     else:
#                         print(f"Could not get 5 distance scores for pair {i//2 + 1}.")
#                         all_comparison_results.append({
#                             'pair_index': i//2 + 1,
#                             'image1_name': temp_image1_file.name,
#                             'image2_name': temp_image2_file.name,
#                             'error': 'Could not get 5 distance scores.'
#                          })
#                 except Exception as e:
#                     print(f"an error occured while processing pair {i//2+1}: {e}")
#                     all_comparison_results.append({
#                         'pair_index': i//2 + 1, # Pair number (starting from 1)
#                         'image1_name': temp_image1_file.name,
#                         'image2_name': temp_image2_file.name,
#                         'error': f"an error occured: {e}"
#                     })
#                 finally:
#                     os.remove(image1_path) if os.path.exists(image_1_path) else None
#                     os.remove(image2_path) if os.path.exists(image_2_path) else None
#             context = {'compareform': compareform, 'results': all_comparison_results}
#             return render(request, 'compareimg.html', context)
#         else:
#             context = {'compareform': compareform}
#             return render(request, 'compareimg.html', context)
#     else:
#         context = {'compareform': compareform}
#         return render(request, 'compareimg.html', context)

    

def video_scanner(request):
    matches = []
    snapshots = []
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            fs = FileSystemStorage()
            filename = fs.save(video.name, video)
            video_path = fs.path(filename)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps * 1)

            frame_num = 0
            snapshot_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % interval == 0:
                    snapshot_path = os.path.join(settings.MEDIA_ROOT, f'snapshot_{snapshot_index}.jpg')
                    cv2.imwrite(snapshot_path, frame)

                    try:
                        results = df.find(img_path=snapshot_path, db_path=os.path.join(settings.MEDIA_ROOT, 'photos'), enforce_detection=False)
                        print("Results for snapshot:", snapshot_path)
                        print("Returned results:", results)


                        if results and not results[0].empty:
                            snapshot_rel_path = f'media/snapshot_{snapshot_index}.jpg'
                            snapshots.append(snapshot_rel_path)

                            match_info = []
                            for i, row in results[0].iterrows():
                                full_path = row['identity']
                                # Convert full path to relative path (e.g., media/photos/image1.jpg)
                                rel_path = os.path.relpath(full_path, settings.BASE_DIR).replace("\\", "/")
                                match_info.append({
                                    'image': rel_path,
                                    'distance': row['distance']
                                })

                            matches.append(match_info)
                        # if results and not results[0].empty:
                        #     print("Number of Matches Found:", len(results[0]))
                        #     matches.append(results[0].to_dict('records'))
                        #     snapshots.append(f'media/snapshot_{snapshot_index}.jpg')
                    except Exception as e:
                        print("DeepFace error:", e)

                    snapshot_index += 1
                frame_num += 1

            cap.release()
            os.remove(video_path)

    context = {
        'matches': matches,
        'snapshots': snapshots,
        'combined': zip(snapshots, matches)
    }

    return render(request, 'videoscanning.html', context)