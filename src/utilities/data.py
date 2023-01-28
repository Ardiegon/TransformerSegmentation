import dlib

def get_dlib_models(weights_path = "checkpoints/shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(weights_path)
    return detector, predictor
    