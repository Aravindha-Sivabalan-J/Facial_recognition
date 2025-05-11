def compare_faces(request):
    all_comparison_results = []
    compareform = CompareFacesForm(request.POST, request.FILES)
    if request.method == 'POST':
        if compareform.is_valid()
        uploaded_files = request.FILES.getlist('images')
        if len(uploaded_files)%2 != 0 or len(uploaded_files)<2:
            context = {'compareform': compareform, 'error': 'Please upload an even number of images (at least two) to form pairs.'}
            return render(request, 'compareimg.html', context)
        for i in range(0, len(uploaded_files), 2):
            image_1 = uploaded_files[i]
            image_2 = uploaded_files[i+1]
            temp_image1_filename = f'temp_upload_{request.user.id or "anon"}_{i}_1_{image_1.name}'
            temp_image2_filename = f'temp_upload_{request.user.id or "anon"}_{i}_2_{image_2.name}'
            image_1_path = os.path.join(settings.MEDIA_ROOT, 'temp_image1_filename')
            image_2_path = os.path.join(settings.MEDIA_ROOT, 'temp_image2_filename')
            try:
                with open(image_1_path, 'wb+')as destination:
                    for chunk in image_1.chunks():
                        destination.write(chunk)
                with open(image_2_path, 'wb+')as destination:
                    for chunk in image_1.chunks():
                        destination.write(chunk)
                distance_scores = []
                try:
                    if image_1_path is not None and image_2_path is not None:
                        dummy1 = df.verify(image_1_path, image_2_path, model_name='Facenet512', enforce_detection=False)
                        distance_scorees.append(dummy1['distance'])
                        dummy2 = df.verify(image_1_path, image_2_path, model_name='Facenet', enforce_detection=False)
                        distance_scorees.append(dummy1['distance'])
                        dummy3 = df.verify(image_1_path, image_2_path, model_name='Dlib', enforce_detection=False)
                        distance_scorees.append(dummy1['distance'])
                        dummy4 = df.verify(image_1_path, image_2_path, model_name='ArcFace', enforce_detection=False)
                        distance_scorees.append(dummy1['distance'])
                        dummy5 = df.verify(image_1_path, image_2_path, model_name='VGG-Face', enforce_detection=False)
                        distance_scorees.append(dummy1['distance'])
                    else:
                        context = {'compareform': compareform, 'error' = 'either one of the file path missing'}
                        return render(request, 'compareimg.html', context)
                except exception as e:
                    print(f"DeepFace verification failed for pair {i//2 + 1}: {e}")
                    all_comparison_results.append({
                            'pair_index': i//2 + 1, # Pair number (starting from 1)
                            'image1_name': image1_file.name,
                            'image2_name': image2_file.name,
                            'error': f'Verification failed: {e}'
                        })
                    continue

                original_distance_array = np.array(distance_scores)

                if original_distace_array.shape[0] == 5:
                    min_distance = np.min(original_distnace_array)
                    max_distance = np.max(original_distance_array)
                    mean_distance = np.mean(original_distance_array)
                    std_distance = np.std(origina_distance_array)

                input_feature_for_prediction = np.hstack(original_distance_array, min_distance, max_distance, mean_distance, std_distance)
                input_feature_for_prediction = input_feature_for_prediction.reshape(1,-1)   

                if loaded_scaler is not None:
                    scaled_input_for_prediction = loaded_scaler.transform(input_feature_for_prediction)
                else:
                    context = {'compareform': compareform, 'error': 'scaler not available.'}
                    return render(request, 'compareimg.html', context)

                if loaded_model is not None:
                    probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
                    probability_of_match_number = probability_of_match_array[0,1]
                else:
                    context = {'compareform': compareform, 'error': 'Model not available.'}
                    return render(request, 'compareimg.html', context)
                chosen_threshold = 0.48
                final_decision = "Not a match"
                if probability_of_match_number >= chosen_threshold:
                    final_decision = "Match"
                    print(final_decision)
                    all_comparison_results.append({
                        "image_index" = i//2+1,
                        "image_1_name" = temp_image1_filename,
                        "image_1_name" = temp_image2_filename,
                        "final_decision" = final_decision,
                    })
                    print(all_comparison_results)
                else:
                    print(f"Could not get 5 distance scores for pair {i//2 + 1}.")
                    all_comparison_results.append({
                        'pair_index': i//2 + 1,
                        'image1_name': image1_file.name,
                        'image2_name': image2_file.name,
                        'error': 'Could not get 5 distance scores.'
                    })
            except exception as e:
                print(f"an error occured while processing pair {i//2+1}: {e}")
                else:
                    all_comparison_results.append({
                        'pair_index': i//2 + 1, # Pair number (starting from 1)
                        'image1_name': image1_file.name,
                        'image2_name': image2_file.name,
                        'error': f"an error occured: {e}"
                    })
            finally:
                os.remove(image1_path) if os.path.exists(image1_path) else None
                os.remove(image2_path) if os.path.exists(image2_path) else None
            context = {'compareform': compareform, 'results': all_comparison_results}
            return render(request, 'compareimg.html', context)
        




def compare_faces(request):
    # Initialize list to store results for ALL pairs
    all_comparison_results = []
    # Bind form with POST data and FILES
    compareform = CompareFacesForm(request.POST, request.FILES)

    # --- Handle POST request (form submission) ---
    if request.method == 'POST':
        # Check if the form data is valid
        if compareform.is_valid():
            # Get the list of uploaded files from the 'images' field
            uploaded_files = request.FILES.getlist('images')

            # Check if we have an even number of files (required for pairs)
            if len(uploaded_files) % 2 != 0 or len(uploaded_files) < 2:
                context = {'compareform': compareform, 'error': 'Please upload an even number of images (at least two) to form pairs.'}
                # Return render immediately if file count is invalid
                return render(request, 'compareimg.html', context)

            # --- Loop through the uploaded files in pairs ---
            # i will be 0, 2, 4, ...
            for i in range(0, len(uploaded_files), 2):
                # Get the two files for the current pair
                image1_file = uploaded_files[i] # Using original names for clarity
                image2_file = uploaded_files[i+1]

                # --- Save uploaded files temporarily for the current pair ---
                # Create unique filenames
                temp_image1_filename = f'temp_upload_{request.user.id if request.user.is_authenticated else "anon"}_{i}_1_{image1_file.name}'
                temp_image2_filename = f'temp_upload_{request.user.id if request.user.is_authenticated else "anon"}_{i}_2_{image2_file.name}'
                # Construct full paths using generated filenames
                image1_path = os.path.join(settings.MEDIA_ROOT, temp_image1_filename)
                image2_path = os.path.join(settings.MEDIA_ROOT, temp_image2_filename)

                # Use a try/except/finally block to process each pair and ensure cleanup
                try:
                    # Write files to temporary paths
                    with open(image1_path, 'wb+') as destination:
                        for chunk in image1_file.chunks():
                            destination.write(chunk)
                    with open(image2_path, 'wb+') as destination:
                         # FIX: Write chunks from image2_file, not image1_file
                        for chunk in image2_file.chunks():
                            destination.write(chunk)

                    # --- Get the 5 original distance scores using DeepFace for the current pair ---
                    distance_scores = []
                    # Use inner try/except for DeepFace specific errors
                    try:
                        # Check paths exist before calling deepface (optional but safe)
                        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
                             raise FileNotFoundError("Temporary file(s) not found.")

                        dummy1 = df.verify(image1_path, image2_path, model_name='Facenet512', enforce_detection=False)
                        distance_scores.append(dummy1['distance'])
                        dummy2 = df.verify(image1_path, image2_path, model_name='Facenet', enforce_detection=False)
                        distance_scores.append(dummy2['distance'])
                        dummy3 = df.verify(image1_path, image2_path, model_name='Dlib', enforce_detection=False)
                        distance_scores.append(dummy3['distance'])
                        dummy4 = df.verify(image1_path, image2_path, model_name='ArcFace', enforce_detection=False)
                        distance_scores.append(dummy4['distance'])
                        dummy5 = df.verify(image1_path, image2_path, model_name='VGG-Face', enforce_detection=False)
                        distance_scores.append(dummy5['distance'])


                    # FIX: Use 'Exception' instead of 'exception'
                    except Exception as e:
                        print(f"DeepFace verification failed for pair {i//2 + 1}: {e}")
                        all_comparison_results.append({
                            'pair_index': i//2 + 1, 
                            'image1_name': image1_file.name,
                            'image2_name': image2_file.name,
                            'error': f'Verification failed: {e}'
                        })
                        continue


                    
                    original_distance_array = np.array(distance_scores)

                    
                    if original_distance_array.shape[0] == 5:

                        
                        min_distance = np.min(original_distance_array)
                        max_distance = np.max(original_distance_array)
                        mean_distance = np.mean(original_distance_array)
                        std_distance = np.std(original_distance_array)

                        input_features_for_prediction = np.hstack((original_distance_array, min_distance, max_distance, mean_distance, std_distance))
                        input_features_for_prediction = input_features_for_prediction.reshape(1, -1) # FIX: Use correct variable name

                        # --- Scale input using the loaded scaler ---
                        if loaded_scaler is not None:
                            scaled_input_for_prediction = loaded_scaler.transform(input_features_for_prediction)
                        else:
                            all_comparison_results.append({
                                'pair_index': i//2 + 1,
                                'image1_name': image1_file.name,
                                'image2_name': image2_file.name,
                                'error': 'Model scaler not available.'
                            })
                            continue


                        
                        if loaded_model is not None:
                    
                            probability_of_match_array = loaded_model.predict_proba(scaled_input_for_prediction)
                            probability_of_match_number = probability_of_match_array[0, 1]
                        else:
                            
                            all_comparison_results.append({
                                'pair_index': i//2 + 1,
                                'image1_name': image1_file.name,
                                'image2_name': image2_file.name,
                                'error': 'Prediction model not available.'
                            })
                            continue

                        # --- Apply the chosen threshold ---
                        chosen_threshold = 0.48
                        final_decision = "Not a Match"
                        if probability_of_match_number >= chosen_threshold:
                            final_decision = "Match"

                        # Append result for THIS pair to the list
                        all_comparison_results.append({
                            'pair_index': i//2 + 1,
                            'image1_name': image1_file.name,
                            'image2_name': image2_file.name,
                            'final_decision': final_decision,
                            'confidence_score': f"{probability_of_match_number:.4f}",
                            'original_distances': original_distance_array.tolist()
                        })

                    # FIX: This else block should be associated with the original_distance_array.shape[0] == 5 check
                    else:
                        print(f"Could not get 5 distance scores for pair {i//2 + 1}.")
                        all_comparison_results.append({
                        'pair_index': i//2 + 1,
                        'image1_name': image1_file.name,
                        'image2_name': image2_file.name,
                        'error': 'Could not get 5 distance scores from DeepFace.'
                        })


                
                except Exception as e:
                    # Catch any other unexpected errors during file saving or processing of this pair
                    print(f"An unexpected error occurred processing pair {i//2 + 1}: {e}")
                    all_comparison_results.append({
                         'pair_index': i//2 + 1,
                         'image1_name': image1_file.name,
                         'image2_name': image2_file.name,
                         'error': f'An unexpected error occurred during processing: {e}'
                    })

                finally:
                    os.remove(image1_path) if os.path.exists(image1_path) else None
                    os.remove(image2_path) if os.path.exists(image2_path) else None

            # --- END OF FOR LOOP ---
            context = {'compareform': compareform, 'results': all_comparison_results}
            return render(request, 'compareimg.html', context)


        
        else:
            context = {'compareform': compareform}
            return render(request, 'compareimg.html', context)


    
    else:
        context = {'compareform': compareform, 'results': all_comparison_results}
        return render(request, 'compareimg.html', context)

                



