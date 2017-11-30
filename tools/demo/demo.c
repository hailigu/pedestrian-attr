
/*
 * @file demo.c
 * @author gongjia
 *
 * FFmpeg push RTMP stream
 *
 * Implements the following command:
 *
 * ffmpeg -re -i input.flv -vcodec copy -acodec copy -f flv -y rtmp://xxxx
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>
#include <assert.h>


#include <libavformat/avformat.h>
#include <libavutil/time.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>

#ifdef NODEBUG
#define LOG(fmt, ...) do {} while (0)
#else
#define LOG(fmt, ...) fprintf(stdout, "[DEBUG] %s:%s:%d: " fmt "\n", __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif


void copy_stream_info(AVStream *ostream, AVStream *istream, AVFormatContext *ofmt_ctx) {
    AVCodecContext *icodec = istream->codec;
    AVCodecContext *ocodec = ostream->codec;

    ostream->id = istream->id;
    ocodec->codec_id = icodec->codec_id;
    ocodec->codec_type = icodec->codec_type;
    ocodec->bit_rate = icodec->bit_rate;

    int extra_size = (uint64_t)icodec->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE;
    ocodec->extradata = (uint8_t *)av_mallocz(extra_size);
    memcpy(ocodec->extradata, icodec->extradata, icodec->extradata_size);
    ocodec->extradata_size = icodec->extradata_size;

    // Some formats want stream headers to be separate.
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        ostream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
    }
}

void copy_video_stream_info(AVStream *ostream, AVStream *istream, AVFormatContext *ofmt_ctx) {
    copy_stream_info(ostream, istream, ofmt_ctx);

    AVCodecContext *icodec = istream->codec;
    AVCodecContext *ocodec = ostream->codec;

    ocodec->width = icodec->width;
    ocodec->height = icodec->height;
    ocodec->time_base = icodec->time_base;
    ocodec->gop_size = icodec->gop_size;
    ocodec->pix_fmt = icodec->pix_fmt;
}

int video2video(const char *in_filename, const char *out_filename, int speed) {
    AVOutputFormat *ofmt = NULL;
    AVFormatContext *ifmt_ctx = NULL; // Input AVFormatContext
    AVFormatContext *ofmt_ctx = NULL; // Output AVFormatContext
    AVDictionaryEntry *tag = NULL;
    AVStream *in_stream = NULL;
    AVStream *out_stream = NULL;
    AVPacket pkt;


    int ret = 0;
    int i = 0;
    int video_index = 0;
    int64_t last_pts = 0;

    memset(&pkt, 0, sizeof(pkt));

    av_register_all();

    // Network
    avformat_network_init();

    // Input
    if ((ret = avformat_open_input(&ifmt_ctx, in_filename, 0, 0)) < 0) {
        LOG("Open input file failed.");
        goto end;
    }

    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
        LOG("Retrieve input stream info failed");
        goto end;
    }

    LOG("ifmt_ctx->nb_streams = %u", ifmt_ctx->nb_streams);

    // Find the first video stream
    video_index = -1;
    for (i = 0; i < ifmt_ctx->nb_streams; ++i) {
        if (AVMEDIA_TYPE_VIDEO == ifmt_ctx->streams[i]->codec->codec_type) {
            video_index = i;
            break;
        }
    }

    if (-1 == video_index) {
        LOG("Didn't find a video stream.");
        goto end;
    }
    LOG("video_index = %d\n", video_index);

    av_dump_format(ifmt_ctx, 0, in_filename, 0);

    // Output
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, out_filename); // RTMP

    if (!ofmt_ctx) {
        LOG("Create output context failed.");
        ret = AVERROR_UNKNOWN;
        goto end;
    }

    ofmt = ofmt_ctx->oformat;

    if ((video_index = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0)) >= 0) {
        in_stream = ifmt_ctx->streams[video_index];
        out_stream = avformat_new_stream(ofmt_ctx, NULL);

        if (!out_stream) {
            LOG("Allocate output stream failed.");
            ret = AVERROR_UNKNOWN;
            goto end;
        }
        copy_video_stream_info(out_stream, in_stream, ofmt_ctx);
    }

    // Copy metadata
    while ((tag = av_dict_get(ifmt_ctx->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
        LOG("%s = %s", tag->key, tag->value);
        av_dict_set(&ofmt_ctx->metadata, tag->key, tag->value, 0);
    }

 

    // Dump Format
    av_dump_format(ofmt_ctx, 0, out_filename, 1);

    // Open output URL
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, out_filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            LOG("Open output URL '%s' failed", out_filename);
            goto end;
        }
    }

    // Write file header
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) {
        LOG("Write output URL failed.");
        goto end;
    }

    av_init_packet(&pkt);

   

    while (1) {
        ret = av_read_frame(ifmt_ctx, &pkt);
        if(ret < 0){
            av_free_packet(&pkt); 
            LOG("");
            break;
        }
        if(pkt.stream_index == video_index){
        
            pkt.pts *= speed;
            pkt.dts *= speed;

            // pkt.pts += last_pts;
            // pkt.dts += last_pts;

            if (av_interleaved_write_frame(ofmt_ctx, &pkt) < 0) {
                LOG("\n");
                break;
            }
            av_free_packet(&pkt);
        }
    }

    av_dump_format(ofmt_ctx, 0, out_filename, 1);
    // Write file trailer
    av_write_trailer(ofmt_ctx);

    end:
        avformat_close_input(&ifmt_ctx);
        /* close output */
        if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE)) {
            avio_close(ofmt_ctx->pb);
        }
        avformat_free_context(ofmt_ctx);
        if (ret < 0 && ret != AVERROR_EOF) {
            LOG( "Error occurred.");
            return -1;
        }

    return 0;
}

int main(int argc, char *argv[]) {
    if(argc<3){
        fprintf(stdout,"usage:./a.out xxx.pcm xxx.aac\n");
        return -1;
    }
    int speed = atoi(argv[3]);
    // ./demo copy.flv rtmp://127.0.0.1/hks/gongjia
    (void)video2video(argv[1], argv[2], speed);

    return 0;
}
