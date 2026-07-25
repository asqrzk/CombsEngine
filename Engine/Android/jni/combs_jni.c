/*
 * combs_jni.c — JNI glue between Kotlin and the combs C ABI (combs.h).
 *
 * Build: `cargo xtask target android-arm64` produces libcombs_ffi.so;
 * copy it to app/src/main/jniLibs/arm64-v8a/ alongside this shim compiled
 * as libcombs_jni.so (see Engine/Android/README.md).
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include "../../../Core/include/combs.h"

/* Stream callback: forwards JSON events to the JVM-side callback object. */
static JavaVM *g_vm;

typedef struct {
    jobject callback; /* global ref to ai.combs.StreamCallback */
    jmethodID onEvent;
} CallbackCtx;

static void stream_cb(const char *event_json, void *user_data) {
    CallbackCtx *ctx = (CallbackCtx *)user_data;
    JNIEnv *env = NULL;
    if (!g_vm || !ctx) return;
    (*g_vm)->AttachCurrentThread(g_vm, &env, NULL);
    if (!env) return;
    jstring json = (*env)->NewStringUTF(env, event_json);
    (*env)->CallVoidMethod(env, ctx->callback, ctx->onEvent, json);
    (*env)->DeleteLocalRef(env, json);
    (*g_vm)->DetachCurrentThread(g_vm);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_vm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT jstring JNICALL
Java_ai_combs_CombsEngine_nativeDeviceCaps(JNIEnv *env, jclass cls) {
    (void)cls;
    char *caps = combs_device_caps_json();
    if (!caps) return NULL;
    jstring out = (*env)->NewStringUTF(env, caps);
    combs_string_free(caps);
    return out;
}

JNIEXPORT jlong JNICALL
Java_ai_combs_CombsEngine_nativeCreate(JNIEnv *env, jclass cls, jstring configJson) {
    (void)cls;
    const char *config = (*env)->GetStringUTFChars(env, configJson, NULL);
    CombsEngine *engine = combs_engine_create(config);
    (*env)->ReleaseStringUTFChars(env, configJson, config);
    return (jlong)engine;
}

JNIEXPORT void JNICALL
Java_ai_combs_CombsEngine_nativeDestroy(JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    combs_engine_destroy((CombsEngine *)handle);
}

JNIEXPORT jstring JNICALL
Java_ai_combs_CombsEngine_nativeMetadata(JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    char *md = combs_engine_metadata_json((CombsEngine *)handle);
    if (!md) return NULL;
    jstring out = (*env)->NewStringUTF(env, md);
    combs_string_free(md);
    return out;
}

JNIEXPORT jint JNICALL
Java_ai_combs_CombsEngine_nativeChatCompletion(
    JNIEnv *env, jclass cls, jlong handle, jstring requestJson, jstring requestId,
    jobject callback) {
    (void)cls;
    CallbackCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.callback = (*env)->NewGlobalRef(env, callback);
    jclass cbClass = (*env)->GetObjectClass(env, callback);
    ctx.onEvent = (*env)->GetMethodID(env, cbClass, "onEvent", "(Ljava/lang/String;)V");

    const char *request = (*env)->GetStringUTFChars(env, requestJson, NULL);
    const char *id = (*env)->GetStringUTFChars(env, requestId, NULL);
    int rc = combs_chat_completion(
        (CombsEngine *)handle, request, id, stream_cb, &ctx);
    (*env)->ReleaseStringUTFChars(env, requestJson, request);
    (*env)->ReleaseStringUTFChars(env, requestId, id);
    (*env)->DeleteGlobalRef(env, ctx.callback);
    return rc;
}

JNIEXPORT jint JNICALL
Java_ai_combs_CombsEngine_nativeCancel(JNIEnv *env, jclass cls, jstring requestId) {
    (void)cls;
    const char *id = (*env)->GetStringUTFChars(env, requestId, NULL);
    int rc = combs_cancel(id);
    (*env)->ReleaseStringUTFChars(env, requestId, id);
    return rc;
}

JNIEXPORT jstring JNICALL
Java_ai_combs_CombsEngine_nativeLastError(JNIEnv *env, jclass cls) {
    (void)cls;
    const char *err = combs_last_error();
    return err ? (*env)->NewStringUTF(env, err) : NULL;
}
