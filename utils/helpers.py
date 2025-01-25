import numpy as np
import re
import os


def list_files_in_directory(directory_path):
    """
    List all files in the given directory.

    Args:
        directory_path (str): Path to the directory.
    """
    if not os.path.isdir(directory_path):
        print(f"The path '{directory_path}' is not a valid directory.")
        return

    return os.listdir(directory_path)


def filter_string(input_string):
    """
    Filters a given string to include only numbers, letters, and underscores.

    Parameters:
        input_string (str): The string to be filtered.

    Returns:
        str: The filtered string containing only numbers, letters, and underscores.
    """
    return re.sub(r'[^\w]', '', input_string)


def filter_alphanumeric(input_string: str) -> str:
    # Use a list comprehension to filter out non-alphanumeric characters
    filtered_string = ''.join(
        [char for char in input_string if char.isalnum()])
    return filtered_string


def cosine_similarity(A, B):
    # Normalize rows of A and B
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    # Compute cosine similarity
    return np.sum(A_norm * B_norm, axis=1)


def cosine_similarity_matrices(A, P):
    """
    Compute the cosine similarity of matrix A with each matrix in P.

    Parameters:
    A : numpy.ndarray
        Matrix of shape (m, n).
    P : numpy.ndarray
        Tensor of shape (p, m, n) containing p matrices to compare with A.

    Returns:
    similarities : numpy.ndarray
        Array of shape (p,) containing cosine similarities of A with each matrix in P.
    """
    # Normalize A
    A_norm = np.linalg.norm(A)
    A_normalized = A / A_norm

    # Normalize each matrix in P
    P_norms = np.linalg.norm(P, axis=(1, 2), keepdims=True)  # Shape (p, 1, 1)
    P_normalized = P / P_norms

    # Compute cosine similarity for each matrix
    dot_products = np.sum(P_normalized * A_normalized,
                          axis=(1, 2))  # Shape (p,)

    return dot_products


def cosine_similarity_batch(A, P_list):
    """
    Compute cosine similarity between rows of matrix A and rows of each matrix in P_list.

    Parameters:
        A: numpy.ndarray of shape (w, m)
        P_list: list of numpy.ndarray, each of shape (n, m)

    Returns:
        similarities: list of numpy.ndarray, each of shape (w, n)
                      where each element contains the cosine similarities
                      between rows of A and rows of the corresponding P matrix.
    """
    # Normalize rows of A
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)

    similarities = []
    for P in P_list:
        # Normalize rows of P
        P_norm = P / np.linalg.norm(P, axis=1, keepdims=True)

        # Compute cosine similarity using matrix multiplication
        sim_matrix = np.dot(A_norm, P_norm.T)  # Shape (w, n)
        similarities.append(sim_matrix)

    return similarities


def top_k_indices(similarities, k):
    """
    Find the indices of the top k largest elements in the array.

    Parameters:
    similarities : numpy.ndarray
        Array of cosine similarity scores.
    k : int
        Number of top indices to return.

    Returns:
    top_indices : numpy.ndarray
        Array of indices corresponding to the top k largest elements.
    """
    # Use NumPy's argpartition to get the top k indices
    partitioned_indices = np.argpartition(-similarities, k)[:k]

    # Sort the top k indices by the actual similarity values in descending order
    top_indices = partitioned_indices[np.argsort(
        -similarities[partitioned_indices])]

    return top_indices


def normalize_rows(matrix):
    """
    Normalize rows of a matrix. Rows with zero norm are left unchanged.

    Parameters:
        matrix: numpy.ndarray - Input matrix

    Returns:
        norm_matrix: numpy.ndarray - Row-normalized matrix
    """
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero: Replace zero norms with 1 (safe division)
    row_norms[row_norms == 0] = 1
    return matrix / row_norms


# def top_k_matches(A, P, k, aggregation='max'):
#     """
#     Find top k matching chunks for the query embedding matrix A, handling cases where w != n.

#     Parameters:
#         A: numpy.ndarray of shape (w, m) - Query embedding matrix
#         P: numpy.ndarray of shape (p, n, m) - Word chunk embeddings (p chunks, each of size n x m)
#         k: int - Number of top matching chunks to return
#         aggregation: str - Aggregation method ('mean' or 'max') for chunk similarity

#     Returns:
#         top_k: list of tuples [(score, chunk_index), ...] - Top k matching chunks
#                Each tuple contains the aggregated similarity score and chunk index.
#     """

#     # Normalize rows of A
#     A_norm = normalize_rows(A)

#     # Normalize rows of each word chunk
#     P_norm = np.array([normalize_rows(chunk) for chunk in P])

#     # Compute cosine similarities
#     similarities = np.einsum('wm,pnm->wpn', A_norm, P_norm)  # Shape (w, p, n)

#     # Aggregate similarity scores for each chunk
#     if aggregation == 'mean':
#         # Average over all query and chunk rows
#         chunk_scores = np.mean(similarities, axis=(0, 2))
#     elif aggregation == 'max':
#         # Max similarity across all rows
#         chunk_scores = np.max(similarities, axis=(0, 2))
#     else:
#         raise ValueError(
#             "Unsupported aggregation method. Use 'mean' or 'max'.")

#     # Find top k chunks
#     top_k_indices = np.argsort(-chunk_scores)[:k]  # Sort in descending order
#     top_k = [(chunk_scores[idx], idx) for idx in top_k_indices]

#     return top_k

def pad_query(query_embedding, target_length):
    """
    Pads the query embedding to the target length.
    Args:
        query_embedding: numpy array of shape (q, d) where q is the number of query words and d is the embedding dimension.
        target_length: int, the desired length after padding.
    Returns:
        Padded numpy array of shape (1, target_length, d).
    """
    q, d = query_embedding.shape
    if q > target_length:
        raise ValueError("Query length exceeds the target padding length.")

    # Pad with zeros to match the target length
    padded_query = np.zeros((1, target_length, d))
    padded_query[0, :q, :] = query_embedding
    return padded_query


def top_k_matches(A, B, top_k=5):
    """
    Computes the top_k matches and corresponding indices between A and B using maximum similarity.
    Args:
        A: numpy array of shape (1, w, d), the query embedding.
        B: numpy array of shape (p, w, d), the document embeddings.
        top_k: int, the number of top matches to return.
    Returns:
        A tuple (top_scores, top_indices) where:
            top_scores: numpy array of shape (top_k,), the top-k similarity scores.
            top_indices: numpy array of shape (top_k,), the indices of the corresponding chunks in B.
    """
    p, w, d = B.shape

    # Compute pairwise cosine similarities between A and B
    # A has shape (1, w, d), B has shape (p, w, d)
    dot_products = np.einsum('nwd,pwd->npw', A, B)  # Shape: (1, p, w)
    A_norms = np.linalg.norm(A, axis=2)  # Shape: (1, w)
    B_norms = np.linalg.norm(B, axis=2)  # Shape: (p, w)

    # Normalize the dot products using norms
    norms = np.expand_dims(A_norms, axis=1) * \
        B_norms[None, :, :]  # Shape: (1, p, w)

    # Avoid division by zero
    norms[norms == 0] = 1e-9
    cosine_similarities = dot_products / norms  # Shape: (1, p, w)

    # Compute the max similarity for each chunk
    max_similarities = np.max(
        cosine_similarities, axis=2).squeeze()  # Shape: (p,)

    # Get the top_k scores and indices
    top_indices = np.argsort(-max_similarities)[:top_k]
    top_scores = max_similarities[top_indices]

    result = [(top_scores[i], top_indices[i]) for i in range(top_k)]

    return result

# print(list_files_in_directory('.ragatouille/colbert/indexes'))
