def video_scanner(request):
    matches = []
    snapshots = []
    if request.method == 'POST':
        if 'video' in request.FILES:
            video = request.FILES['video']
            fs = FileSystemStorage()
            filename = fs.save(video.name, video)
            video_path = fs.path(filename)

            unique = set()
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
                print(f"Frame {current_frame_index}: ret={ret}")
                frame_num += 1
                if not ret:
                    break
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray_frame)
                if mean_intensity<BLACK_FRAME_THRESHOLD:
                    print(f"skipping {current_frame_index} as it is too dark/black")
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
                            enforce_detection=False)
                        print("Results for snapshot:", snapshot_path)
                        print("Returned results:", results)


                        if results and not results[0].empty:
                            confirm_match = []
                            similar_match = []
                            custom_threshold = 0.5
                            loose_threshold = 0.61

                            has_new_person_for_display = False

                            for i, row in results[0].iterrows():
                                if row['distance']<=custom_threshold:
                                    full_path = row['identity']
                                    rel_path = os.path.relpath(full_path, settings.BASE_DIR).replace("\\", "/")
                                    confirm_match.append({
                                        'image': rel_path,
                                        'distance': row['distance'],
                                        'status': 'confirmed match'
                                    })
                                    if row['identity'] not in unique:
                                        has_new_person_for_display = True
                                elif row['distance']<=loose_threshold:
                                    full_path = row['identity']
                                    rel_path = os.path.relpath(full_path, settings.BASE_DIR).replace("\\","/")
                                    similar_match.append({
                                        'image': rel_path,
                                        'distance': row['distance'],
                                        'status': 'similar profile'
                                    })
                                    if row['identity'] not in unique_persons:
                                        has_new_person_for_display = True
                                else:
                                    print(f"Skipping match for {row['identity']} due to high distance: {row['distance']}")
                            if has_new_person_for_display:
                                for match_item in confirm_match:
                                    unique.add(os.path.join(settings.BASE_DIR, match_item['image']).replace("/", os.sep))
                                for match_item in simialr_match:
                                    unique.add(os.path.join(settings.BASE_DIR, match_item['image']).replace("/", os.sep))
                                snapshot_rel_path = f'media/snapshot_{snapshot_index}.jpg'
                                snapshots.append(snapshot_rel_path)
                                if confirm_match:
                                    matches.append(confirm_match)
                                    print(f"there was a confirm match for snapshot: {snapshot_index}")
                                elif similar_match:
                                    matches.append(similar_match)
                                    print(f"there was a similar match for snapshot: {snapshot_index}")
                                else:
                                    print(f"no valid match for snapshot: {snapshot_index} after applying custom threshold")
                            else:
                                print(f"Snapshot {snapshot_index}: No new unique persons found. Skipping adding to final display.")
                        else:
                            print(f"No faces detected or no matches returned by DeepFace for snapshot: {snapshot_index}")
                    except Exception as e:
                        print("DeepFace error:", e)

                        snapshot_index += 1

                cap.release()
                os.remove(video_path)

    context = {
        'matches': matches,
        'snapshots': snapshots,
        'combined': zip(snapshots, matches)
    }

    return render(request, 'videoscanning.html', context)

from docx import Document
import re

doc = Document(r"C:\Users\aathi\Desktop\newsamplenegative.docx")

pattern = r"Distance Scores\s*:\s*\[[^\]]+\]"

distance_scores_list = []

for para in doc.paragraphs:
    match = re.search(pattern, para.text)
    if match:
        distance_scores_list.append(match.group())

# Print or save the results
for item in distance_scores_list:
    print(item)

# def compare_faces(request):
#     all_comparison_results = []
#     compareform = CompareFacesForm() # Initialize an empty form for GET requests

#     if request.method == 'POST':
#         print("step 1 passed")
#         # Re-initialize form with submitted data for validation within the POST block
#         compareform = CompareFacesForm(request.POST, request.FILES)

#         # <<< CHANGE: Only process if form is valid >>>
#         if compareform.is_valid():

#             # <<< CHANGE: Get list of files using getlist >>>
#             uploaded_files = request.FILES.getlist('images[]')

#             # Check if we have an even number of files (required for pairs)
#             if len(uploaded_files) % 2 != 0 or len(uploaded_files) < 2:
#                 print("uploaded length is not an even number or less than 2")
#                 # <<< CHANGE: Return HTML with error for incorrect file count >>>
#                 context = {'compareform': compareform, 'error': 'Please upload an even number of images (at least two) to form pairs.'}
#                 return render(request, 'compareimg.html', context)

#             # Loop through the uploaded files in pairs
#             for i in range(0, len(uploaded_files), 2):
#                 print(f"Processing pair starting at index {i}")

#                 # Get the two files for the current pair
#                 image_1 = uploaded_files[i]
#                 image_2 = uploaded_files[i+1]

#                 # --- Save uploaded files temporarily for the current pair ---
#                 temp_image1_filename = f'temp_upload_{request.user.id or "anon"}_{i}_1_{image_1.name}'
#                 temp_image2_filename = f'temp_upload_{request.user.id or "anon"}_{i}_2_{image_2.name}'

#                 # Ensure MEDIA_ROOT path exists
#                 media_root_path = settings.MEDIA_ROOT
#                 if not os.path.exists(media_root_path):
#                     os.makedirs(media_root_path)

#                 image_1_path = os.path.join(media_root_path, temp_image1_filename)
#                 image_2_path = os.path.join(media_root_path, temp_image2_filename)

#                 # <<< CHANGE: Outer try/except block for saving and processing >>>
#                 try:
#                     # Write files to temporary paths
#                     with open(image_1_path, 'wb+') as destination:
#                         for chunk in image_1.chunks():
#                             destination.write(chunk)
#                     with open(image_2_path, 'wb+') as destination:
#                          # <<< CHANGE: Corrected bug - write chunks from image_2 >>>
#                         for chunk in image_2.chunks():
#                             destination.write(chunk)

#                     distance_scores = []
#                     # Inner try/except specifically for DeepFace verification
#                     try:
#                         dummy1 = df.verify(image_1_path, image_2_path, model_name='Facenet512', enforce_detection=False)
#                         distance_scores.append(dummy1['distance'])
#                         dummy2 = df.verify(image_1_path, image_2_path, model_name='Facenet', enforce_detection=False)
#                          # <<< CHANGE: Corrected bug - append dummy2['distance'] >>>
#                         distance_scores.append(dummy2['distance'])
#                         dummy3 = df.verify(image_1_path, image_2_path, model_name='Dlib', enforce_detection=False)
#                          # <<< CHANGE: Corrected bug - append dummy3['distance'] >>>
#                         distance_scores.append(dummy3['distance'])
#                          # <<< CHANGE: Corrected bug - ensure model names are consistent >>>
#                         dummy4 = df.verify(image_1_path, image_2_path, model_name='VGG-Face', enforce_detection=False)
#                          # <<< CHANGE: Corrected bug - append dummy4['distance'] >>>
#                         distance_scores.append(dummy4['distance'])
#                          # <<< CHANGE: Corrected bug - ensure model names are consistent >>>
#                         dummy5 = df.verify(image_1_path, image_2_path, model_name='ArcFace', enforce_detection=False)
#                          # <<< CHANGE: Corrected bug - append dummy5['distance'] >>>
#                         distance_scores.append(dummy5['distance'])

#                     # Catch exceptions specifically from DeepFace verification
#                     except Exception as e:
#                         print(f"DeepFace verification failed for pair {i//2 + 1} ({image_1.name}, {image_2.name}): {e}")
#                         # Add an error result for this pair
#                         all_comparison_results.append({
#                             'pair_index': i//2 + 1,
#                             'image1_name': image_1.name,
#                             'image2_name': image_2.name,
#                             'error': f'Verification failed: {e}'
#                         })
#                         # Skip to the next pair in the loop
#                         continue # Jumps to the next iteration of the for loop

#                     # Convert distances to numpy array
#                     original_distance_array = np.array(distance_scores)

#                     # Check if we got exactly 5 distances
#                     if original_distance_array.shape[0] == 5:
#                          # Calculate the 4 new features (min, max, mean, std)
#                         min_distance = np.min(original_distance_array)
#                         max_distance = np.max(original_distance_array)
#                         mean_distance = np.mean(original_distance_array)
#                         std_distance = np.std(original_distance_array)

#                         # Combine features and reshape
#                         # <<< CHANGE: np.hstack needs args in a tuple >>>
#                         input_features_for_prediction = np.hstack((original_distance_array, min_distance, max_distance, mean_distance, std_distance))
#                         input_features_for_prediction = input_features_for_prediction.reshape(1,-1)

#                         # Scale input using the loaded scaler
#                         if loaded_scaler is not None:
#                             scaled_input_for_prediction = loaded_scaler.transform(input_features_for_prediction)
#                         else:
#                             # If scaler not loaded, return error immediately (cannot process any pairs)
#                             # <<< CHANGE: Return JsonResponse with error if scaler missing >>>
#                             return JsonResponse({'error': 'Model scaler not available.'}, status=500)


#                         # Predict probability using the loaded model
#                         if loaded_model is not None:
#                             probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
#                             probability_of_match_number = probability_of_match_array[0,1]
#                         else:
#                              # If model not loaded, return error immediately
#                             # <<< CHANGE: Return JsonResponse with error if model missing >>>
#                             return JsonResponse({'error': 'Prediction model not available.'}, status=500)


#                         # Apply the chosen threshold
#                         chosen_threshold = 0.499 # Use your best threshold
#                         final_decision = "Not a match"
#                         if probability_of_match_number >= chosen_threshold:
#                             final_decision = "Match"
#                         # print(final_decision) # For debugging

#                         # <<< CHANGE: Append result for THIS pair to the list >>>
#                         all_comparison_results.append({
#                             "pair_index": i//2+1, # Pair number (starting from 1)
#                             "image1_name": image_1.name, # Use original file names
#                             "image2_name": image_2.name, # Use original file names
#                             "final_decision": final_decision,
#                             'confidence_score': f"{probability_of_match_number:.4f}",
#                             'original_distances': original_distance_array.tolist()
#                         })
#                         print(f"Finished pair {i//2 + 1}") # Keep your debugging prints

#                     # <<< CHANGE: Correctly place the else block for the 5-distance check >>>
#                     else: # This else corresponds to 'if original_distance_array.shape[0] == 5:'
#                         print(f"Could not get 5 distance scores for pair {i//2 + 1} ({image_1.name}, {image_2.name}).")
#                         all_comparison_results.append({
#                            'pair_index': i//2 + 1,
#                            'image1_name': image_1.name,
#                            'image2_name': image_2.name,
#                            'error': 'Could not get 5 distance scores.'
#                         })

#                 # <<< CHANGE: Outer except block for errors during saving or general processing of this pair >>>
#                 except Exception as e:
#                     print(f"An unexpected error occurred processing pair {i//2+1} ({image_1.name}, {image_2.name}): {e}")
#                     all_comparison_results.append({
#                         'pair_index': i//2 + 1, # Pair number (starting from 1)
#                         'image1_name': image_1.name,
#                         'image2_name': image_2.name,
#                         'error': f"An unexpected error occurred: {e}"
#                     })

#                 finally:
#                     # --- Clean up temporary files for the CURRENT pair regardless of outcome ---
#                      # <<< CHANGE: Use the correct path variables for os.remove and os.path.exists >>>
#                     os.remove(image_1_path) if os.path.exists(image_1_path) else None
#                     os.remove(image_2_path) if os.path.exists(image_2_path) else None
#                     print(f"Cleaned up temp files for pair {i//2 + 1}")


#             # <<< MAJOR CHANGE: Return JsonResponse AFTER the for loop finishes >>>
#             # This ensures all pairs are processed before sending the final response
#             # <<< CHANGE: Return JsonResponse with the list of all results >>>
#             return JsonResponse({'results': all_comparison_results})


#         # <<< CHANGE: If form is NOT valid (e.g., no files uploaded), return a JsonResponse with errors >>>
#         else:
#             # Form is not valid, return JSON response with form errors
#             # <<< CHANGE: Return JsonResponse with form errors >>>
#             return JsonResponse({'error': 'Invalid form submission', 'form_errors': compareform.errors}, status=400) # Return 400 status for bad request


#     # If the request method is GET (user just visited the page)
#     else:
#         # Render the initial empty form page (This part still renders HTML)
#         # <<< CHANGE: Ensure form is initialized for GET and pass empty results list >>>
#         context = {'compareform': compareform, 'results': all_comparison_results} # Pass empty results list
#         # <<< CHANGE: Return render here for GET requests >>>
#         return render(request, 'compareimg.html', context)


# <script>
#   // <<< REMOVED: The console.log from the top - csrftoken was not defined here >>>
#   // console.log('CSRF Token read by JS:', csrftoken);

#   // <<< CHANGE: Moved getCookie function definition OUTSIDE the event listener >>>
#   function getCookie(name) {
#       let cookieValue = null;
#       if (document.cookie && document.cookie !== '') {
#           const cookies = document.cookie.split(';');
#           for (let i = 0; i < cookies.length; i++) {
#               const cookie = cookies[i].trim();
#               // Does this cookie string begin with the name we want?
#               if (cookie.startsWith(name + '=')) {
#                   cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
#                   break;
#               }
#           }
#       }
#       return cookieValue;
#   }
#   // <<< END OF MOVED FUNCTION >>>


#   const input = document.getElementById('image-input');
#   const container = document.getElementById('preview-container');
#   const form = document.getElementById('image-form');

#   let filesArray = [];

#   input.addEventListener('change', () => {
#     container.innerHTML = '';
#     filesArray = [];

#     const seenFiles = new Set();

#     Array.from(input.files).forEach((file, index) => {
#       if (!file.type.startsWith('image/')) return;

#       // Skip duplicates by name + size
#       const uniqueKey = `${file.name}_${file.size}`;
#       if (seenFiles.has(uniqueKey)) return;
#       seenFiles.add(uniqueKey);

#       const reader = new FileReader();
#       reader.onload = function (e) {
#         const box = document.createElement('div');
#         box.className = 'img-box';
#         box.dataset.index = index;

#         const img = document.createElement('img');
#         img.src = e.target.result;
#         box.appendChild(img);

#         box.addEventListener('click', () => {
#           box.classList.toggle('selected');
#         });

#         container.appendChild(box);
#         filesArray.push(file); // Add only once
#       };
#       reader.readAsDataURL(file);
#     });
#   });

#   form.addEventListener('submit', function (e) {
#     e.preventDefault();

#     const selectedBoxes = container.querySelectorAll('.img-box.selected');
#     const selectedIndexes = Array.from(selectedBoxes).map(div => div.dataset.index);

#     const formData = new FormData();
#     selectedIndexes.forEach(i => {
#       formData.append('images[]', filesArray[i]);
#     });

    
#     const csrftoken = getCookie('csrftoken');

   
#     console.log('CSRF Token read by JS (inside submit):', csrftoken);

#     // Post to backend (update '/upload' to your Django URL)
#     fetch('/compareurl/', { // Use the correct URL
#       method: 'POST',
#       body: formData,
#       headers: {
#         'X-CSRFToken': csrftoken // Include the CSRF token in the header
#       }
#     })
#     .then(res => {
#       if (res.ok) {
#            // If the response is OK (e.g., 200 status code)
#            // You might expect a JSON response from your view with results
#            return res.json(); // Assuming your view returns JSON results
#       } else {
#           // If the response is not OK (e.g., 400, 500 status code)
#           // Read the error response body for more details
#           return res.text().then(text => { throw new Error(`Upload failed with status ${res.status}: ${text}`); });
#       }
#     })
#     .then(data => {
#         // This block runs if the fetch and initial response are OK, and JSON is parsed
#         alert("Images uploaded successfully!"); // You will likely want to display results here instead
#         console.log('Comparison results:', data); // Log the results received from the view
#         // *** Now update your HTML to display the data received from the backend ***
#         // This 'data' variable should contain the all_comparison_results list from your view if it returns JSON
#     })
#     .catch(err => {
#       // This block runs if the fetch itself fails or if the response status is not OK and an error was thrown
#       alert("Error uploading images"); // This alert will show on any failure
#       console.error("Fetch or Upload Error:", err); // Log the detailed error
#     });
#   });
# </script>





























